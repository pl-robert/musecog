import os
import shutil
import pickle as pkl
import numpy as np
import pandas as pd

import torch

from utils.midi_processing import convert_midi_to_piano_roll
from dataloader import PianoRollDataset

def joint_entropy(probs, axis = 0, eps = 1e-10):
    '''
    description: entropy (in bits) of the model's predictions over configurations of simultaneous notes (no onset, single notes, chords).
        It ranges from 0 (complete certainty) to 1 bit per pitch (88 bits if all p_i = 0.5).

    parameter: probs, array of predicted probabilities (values in [0, 1])
    parameter: axis, axis of probs corresponding to pitches
    parameter: eps, probabilities are clipped to [eps, 1-eps] to avoid log(0)

    return: entropy (bits), with the pitch axis removed
    '''
    probs = np.clip(probs, eps, 1 - eps)
    return np.sum(-probs*np.log2(probs) - (1-probs)*np.log2(1-probs), axis = axis)

def split_surprise(probs, targets, axis = 0, eps = 1e-10):
    '''
    description: split the surprise (information content of the observed configuration, in bits) into a timing and a pitch part.
        At a timestep with note onsets: -log2 P(configuration) = -log2 P(any onset) + -log2 P(configuration | any onset)
                                                                   (timing surprise)     (pitch surprise)
        where P(any onset) = 1 - prod_i (1 - p_i). At a timestep without onsets, the whole surprise is about timing (that no note
        started) and the pitch surprise is 0. The two parts sum to the 'surprise' feature (binary cross-entropy summed over pitches).

    parameter: probs, array of predicted probabilities (values in [0, 1])
    parameter: targets, binary array of the same shape (1 for pitches with an onset)
    parameter: axis, axis of probs and targets corresponding to pitches
    parameter: eps, probabilities are clipped to [eps, 1-eps] to avoid log(0)

    return: surprise_timing, surprise_pitch (bits), with the pitch axis removed
    '''
    probs = np.clip(probs, eps, 1 - eps)
    surprise = np.sum(np.where(targets == 1, -np.log2(probs), -np.log2(1 - probs)), axis = axis)
    log_p_no_onset = np.sum(np.log1p(-probs), axis = axis)       #natural log of P(no onset) = prod_i (1 - p_i)
    has_onset = np.any(targets == 1, axis = axis)
    surprise_timing = np.where(has_onset, -np.log2(-np.expm1(log_p_no_onset)), -log_p_no_onset / np.log(2))
    surprise_pitch = np.where(has_onset, surprise - surprise_timing, 0)
    return surprise_timing, surprise_pitch

def split_uncertainty(probs, axis = 0, eps = 1e-10):
    '''
    description: split the uncertainty (entropy over configurations of simultaneous notes, see joint_entropy()) into a timing and
        a pitch part, by asking two questions one after the other: "does any note start at this timestep?" (when) and, if so,
        "which set of pitches?" (what). With q = P(any onset) = 1 - prod_i (1 - p_i), the chain rule of entropy gives:
            uncertainty = H_b(q) + q * H(configuration | any onset)
        where H_b(q) = -q*log2(q) - (1-q)*log2(1-q) is the entropy of the yes/no question (the 'no onset' branch adds nothing, since
        the configuration is then known: no pitch starts).

    parameter: probs, array of predicted probabilities (values in [0, 1])
    parameter: axis, axis of probs corresponding to pitches
    parameter: eps, probabilities are clipped to [eps, 1-eps] to avoid log(0)

    return: uncertainty_timing, uncertainty_pitch_conditional, uncertainty_pitch_weighted (bits), with the pitch axis removed
    '''
    probs = np.clip(probs, eps, 1 - eps)
    log_p_no_onset = np.sum(np.log1p(-probs), axis = axis)       #natural log of P(no onset) = prod_i (1 - p_i)
    q = -np.expm1(log_p_no_onset)                                #P(any onset)

    #uncertainty_timing: how uncertain the model is about WHETHER a note starts at this timestep (ignoring which one).
    #Entropy of the yes/no event 'any onset', from 0 (sure that something starts, or sure that nothing starts) to 1 bit (q = 0.5).
    #Note: q is the probability of an onset within one model timestep (1/fs), so this value depends on the model's sampling frequency.
    uncertainty_timing = -q*np.log2(q) - (1 - q)*log_p_no_onset/np.log(2)

    #uncertainty_pitch_weighted: part of the total uncertainty that is about WHICH pitches start, i.e. q * H(configuration | any onset).
    #It is small when the model expects no onset (q close to 0), whatever it would expect if something did start.
    #uncertainty_timing + uncertainty_pitch_weighted = uncertainty (like surprise_timing + surprise_pitch = surprise).
    uncertainty_pitch_weighted = np.maximum(joint_entropy(probs, axis = axis, eps = eps) - uncertainty_timing, 0) #max: rounding errors

    #uncertainty_pitch_conditional: how uncertain the model is about WHICH pitches start, assuming that some notes start:
    #H(configuration | any onset), the entropy of the joint probabilities of all non-empty configurations, renormalized to sum to 1.
    #Counterpart of surprise_pitch (also conditioned on an onset). It can be large when q is small: it then describes an event the
    #model considers unlikely.
    uncertainty_pitch_conditional = uncertainty_pitch_weighted / q

    return uncertainty_timing, uncertainty_pitch_conditional, uncertainty_pitch_weighted

def load_model(model_name, device):
    '''
    description: load a trained model and its information from ./versions/model_name/ (the model class must be imported)

    parameter: model_name, name of the model folder in ./versions/
    parameter: device

    return: model, info (dictionary of model information: model_type, fs, ons_value, sus_value, ...)
    '''
    model = torch.load('./versions/' + model_name + '/model.pt', weights_only=False, map_location=device)
    model.device = device
    info = torch.load('./versions/' + model_name + '/info.pt', map_location=device)
    return model, info

def load_piano_rolls(data_path, info, device, tmp_dir = './tmp_export/'):
    '''
    description: convert all midi files of a folder to piano rolls, in the format expected by a model (sampling frequency,
        onset and sustain values). The files are converted in a temporary folder, which is deleted once the piano rolls are loaded.

    parameter: data_path, path to the folder containing the midi files
    parameter: info, dictionary of model information (from load_model())
    parameter: device
    parameter: tmp_dir, temporary folder for the conversion

    return: samples, list of piano rolls (tensors, time x 88, with info['ons_value'] at onsets and info['sus_value'] at sustains)
    return: file_names, list of file names (without extension)
    return: onset_times, dict {file_name: exact onset times (s) of the notes in the midi file}
    '''
    #create tmp folder (emptied first, in case a previous export was interrupted)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    #convert midi to piano roll
    convert_midi_to_piano_roll(data_path = data_path, out_dir = tmp_dir, file_name = None,
                               fs = info['fs'], pedal_threshold = 64, save_midi_timings = True)

    #load piano rolls
    dataset = PianoRollDataset(data_path = tmp_dir,
               dataset_fs = info['fs'], model_fs = info['fs'],
               ons_value = info['ons_value'], sus_value = info['sus_value'],
               padding_value = info['padding_value'],
               source_length = None,
               use_transposition = False,
               preload = True, device = device, dtype = torch.float32)
    samples, file_names = [], []
    for i_sample in range(dataset.__len__()):
        sample, file_name = dataset.__getitem__(i_sample)
        samples.append(sample)
        file_names.append(file_name.split('.')[0])
    with open(tmp_dir + 'timings.pkl', 'rb') as f:
        onset_times = pkl.load(f)

    #clean tmp folder with its content
    shutil.rmtree(tmp_dir)
    return samples, file_names, onset_times

def predict_piano_roll(model, sample, info, device, silence_duration = 10, return_layers = False):
    '''
    description: compute the model's predictions over a whole piano roll, timestep by timestep.
        A silence is added before the piano roll to avoid a starting bias: the first prediction is made after the model has
        listened to this silence only. The model never sees the last timestep of the piano roll, as there is nothing left to predict.

    parameter: model, trained model (transformer or lstm)
    parameter: sample, piano roll (time x 88) with ons_value at note onsets and sus_value at note sustains
    parameter: info, dictionary of model information (from ./versions/model_name/info.pt)
    parameter: device
    parameter: silence_duration, duration (s) of the silence added before the piano roll
    parameter: return_layers, if True, also return the activations of the model's hidden layers

    return: output, tensor (time x 88). output[t] is the predicted probability of an onset for each pitch at timestep t,
                    knowing the timesteps before t (sample[:t]).
    return: tgt, tensor (time x 88). tgt[t] is 1 for pitches with an onset at timestep t, 0 otherwise.
    return: layers (only if return_layers), dict {layer name: tensor (time x units)}. layers[name][t] is the predictive state that
                    produced output[t]: it has integrated sample[:t], but not sample[t].
                    - transformer: output of each encoder layer ('layer_0', 'layer_1', ...)
                    - lstm: hidden state of the last LSTM layer ('layer_<n_lstm_layers-1>')
    '''
    fs = info['fs']
    silence_length = int(silence_duration*fs)
    sample = sample.to(device)

    #model input: silence followed by the piano roll, without its last timestep
    src = torch.cat((torch.zeros((silence_length, 88), device = device), sample[:-1, :]), dim = 0)

    #targets: note onsets of the piano roll (sustains are not predicted)
    tgt = sample.clone()
    tgt[tgt == info['sus_value']] = 0
    tgt[tgt == info['ons_value']] = 1

    #record the hidden layers' activations during the forward pass (forward hooks)
    captured = {}
    hooks = []
    if return_layers:
        if info['model_type'] == 'transformer':
            for i_layer, layer in enumerate(model.transformer_encoder.layers):
                hooks.append(layer.register_forward_hook(
                    lambda module, inputs, out, name = f'layer_{i_layer}': captured.__setitem__(name, out.squeeze(0))))
        elif info['model_type'] == 'lstm':
            name = f'layer_{model.n_lstm_layers - 1}'
            hooks.append(model.lstm.register_forward_hook( #the lstm output is a packed sequence of the last layer's hidden states
                lambda module, inputs, out, name = name: captured.__setitem__(name,torch.nn.utils.rnn.pad_packed_sequence(out[0], batch_first=True)[0].squeeze(0))))

    #get model predictions, output[i] being the prediction about src[i+1]
    model.eval()
    layers = {}
    with torch.no_grad():
        if info['model_type'] == 'transformer':
            max_seq_length = info['max_seq_length']
            if src.size(0) <= max_seq_length:
                output = model(src.unsqueeze(0)).squeeze(0)
                layers = dict(captured)
            else: #sliding window over sequences longer than the model's context
                output = torch.zeros((src.size(0), 88), device = device)
                output[:max_seq_length, :] = model(src[:max_seq_length, :].unsqueeze(0)).squeeze(0)
                for name, act in captured.items():
                    layers[name] = torch.zeros((src.size(0), act.size(-1)), device = device)
                    layers[name][:max_seq_length, :] = act
                for i in range(1, src.size(0) - max_seq_length + 1):
                    output[i+max_seq_length-1, :] = model(src[i:i+max_seq_length, :].unsqueeze(0)).squeeze(0)[-1, :]
                    for name, act in captured.items():
                        layers[name][i+max_seq_length-1, :] = act[-1, :]
        elif info['model_type'] == 'lstm':
            (h,c) = model.init_hidden(batch_size = 1)
            output, (h,c) = model(src.unsqueeze(0), (h,c))
            output = output.squeeze(0)
            layers = dict(captured)
    for hook in hooks:
        hook.remove()

    #remove initial silence: the prediction about sample[0] is the one made after the last timestep of silence
    output = output[silence_length-1:, :]
    if return_layers:
        layers = {name: act[silence_length-1:, :] for name, act in layers.items()}
        return output, tgt, layers
    return output, tgt

def upsample_features(features, onset_mask, fs, out_fs, discrete_features = [], onset_times = None):
    '''
    description: resample time-resolved features from the model's sampling frequency (fs) to out_fs.
        - continuous features (defined at every timestep, e.g. surprise) are linearly interpolated.
        - discrete features (only defined at timesteps with note onsets, e.g. surprise_positive) keep one value per event, placed at
          the nearest timestep of the new time axis, and are zero elsewhere. A binary 'onset' mask is added in the same way.
        Timing correction: the piano roll places each onset at the nearest model timestep, which shifts it by up to 1/(2*fs). If the
        exact onset times are given, each timestep with onsets is moved to the average time of its onsets before resampling.
        out_fs should be >= fs: otherwise, events closer than 1/out_fs land on the same new timestep and overwrite each other.

    parameter: features, dict of 1D arrays (one value per model timestep)
    parameter: onset_mask, boolean 1D array, True at the model timesteps with note onsets
    parameter: fs, sampling frequency of the model (Hz)
    parameter: out_fs, sampling frequency of the resampled features (Hz)
    parameter: discrete_features, names of the features to treat as discrete (the others are continuous)
    parameter: onset_times, exact onset times (s) of the notes in the midi file, for timing correction (None for no correction)

    return: dict with 'time_axis' (s), the resampled features (same order as in features) and 'onset' (binary mask)
    '''
    event_idx = np.where(onset_mask)[0] #timesteps with note onsets

    #time axis of the model (timestep t at t/fs, as in the piano roll) and new time axis (timestep t at t/out_fs)
    model_time_axis = np.arange(len(onset_mask)) / fs
    new_time_axis = np.arange(int(np.floor(model_time_axis[-1]*out_fs + 1e-6)) + 1) / out_fs

    #timing correction: move each timestep with note onsets to the exact (average) time of its onsets in the midi file
    if onset_times is not None:
        onset_times = np.asarray(onset_times)
        onset_steps = np.round(onset_times*fs).astype(int) #timestep of each onset in the piano roll (same rounding as onset_inst_piano_roll)
        for idx in event_idx:
            if np.any(onset_steps == idx):
                model_time_axis[idx] = np.mean(onset_times[onset_steps == idx])

    #position of each event on the new time axis, for discrete features
    new_event_idx = np.clip(np.round(model_time_axis[event_idx]*out_fs).astype(int), 0, len(new_time_axis)-1)

    upsampled = {'time_axis': new_time_axis}
    for name, values in list(features.items()) + [('onset', onset_mask.astype(float))]:
        if name in discrete_features or name == 'onset':
            upsampled[name] = np.zeros(new_time_axis.shape)
            upsampled[name][new_event_idx] = values[event_idx]
        else:
            upsampled[name] = np.interp(new_time_axis, model_time_axis, values)
    return upsampled

def export_features(data_path, output_path, model_name = 'transformer_maestro_test', out_fs = 100, timing_correction = False):
    '''
    description: export time-resolved features from a trained model. The features are computed from the model's predictions on the input data.

    parameter: data_path, path to the MIDI file from which to retrieve the model's features
    parameter: output_path, path where the features will be stored
    parameter: model_name, name of the model folder in ./versions/
    parameter: out_fs, sampling frequency of the time resolved features. If out_fs is higher than the model's sampling frequency,
    the features will be interpolated.
    parameter: timing_correction. 
        If False, the time resolved features will remain temporally aligned with the piano rolls processed by the model. This will cause 
        small temporal shifts between the features and the actual note onsets, proportional to the model's sampling frequency. 
        If True, the time resolved features will be re-aligned with the exact note onsets, but this may slightly distord the 
        values (linear interpolation). This is useful when precise timing information is required.

    features list:
        Features tagged [TENTATIVE] are ongoing tests: they have not been used in published work yet, and may change or be removed.

        - surprise: Binary Cross Entropy (BCE) between the model's predictions and the target values
        - surprise_positive: positive part of the BCE (pitches with an onset). Only defined at timesteps with onsets.
        - surprise_negative: negative part of the BCE (pitches without an onset)
        - surprise_timing [TENTATIVE]: part of the surprise about when notes start: -log2 P(any onset) at timesteps with onsets,
                       -log2 P(no onset) otherwise. See split_surprise().
        - surprise_pitch [TENTATIVE]: part of the surprise about which pitches start, knowing that some notes start:
                       -log2 P(configuration | any onset). Only defined at timesteps with onsets. surprise = surprise_timing + surprise_pitch.
        - uncertainty [TENTATIVE]: entropy of the model's predictions over configurations of simultaneous notes (no onset, single notes,
                       chords), see joint_entropy(). This is the expected value of 'surprise' under the model's predictions.
        - uncertainty_timing [TENTATIVE]: part of the uncertainty about whether any note starts at this timestep (0 to 1 bit).
                       See split_uncertainty().
        - uncertainty_pitch_weighted [TENTATIVE]: part of the uncertainty about which pitches start, weighted by the probability that
                       some notes start. uncertainty = uncertainty_timing + uncertainty_pitch_weighted.
        - uncertainty_pitch_conditional [TENTATIVE]: uncertainty about which pitches start, assuming that some notes start
                       (= uncertainty_pitch_weighted / P(any onset)).
        - uncertainty_legacy: first approximation of the model's uncertainty, kept for compatibility with earlier analyses. The pitch
                       probabilities are normalized (sum = 1) and treated as a distribution over single notes, which ignores silence
                       and simultaneous notes.
        - predicted_density: predicted note density (sum of probabilities)

    files:
        - features.csv: contains a summary of the features of all midi files in a single table. Each feature is summed over time, either
                        across all timesteps ('continuous_xxx' features) or only at the timesteps with note onsets ('onsets_xxx' features). In
                        addition, the table contains the number of notes, events (group of simulatenous notes) and the duration of each file.
        - features_over_time: folder containing the time-resolved features for each midi file. Each file contain all features, with the 
                        addition of a time axis and a binary mask for the note onsets.
    '''

    #set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    #create output folder
    output_path = output_path + '/features_' + model_name + '/'
    os.makedirs(output_path, exist_ok=True)

    #model import
    model, info = load_model(model_name, device)
    fs = info['fs']
    if out_fs < fs:
        print(f'Warning: out_fs ({out_fs} Hz) is lower than the model sampling frequency ({fs} Hz). '
              'Onset-related features of events closer than 1/out_fs will overwrite each other.')

    #convert midi files to piano rolls
    samples, file_list, timings = load_piano_rolls(data_path, info, device)

    #get model predictions
    output_list = []
    target_list = []
    for i_sample, (sample, file_name) in enumerate(zip(samples, file_list)):
        print('Computing predictions for sample ', i_sample+1, '/', len(samples), ': ', file_name)
        output, tgt = predict_piano_roll(model, sample, info, device)
        output_list.append(output.cpu().numpy().T)
        target_list.append(tgt.cpu().numpy().T)

    #compute features
    features = {}
    onset_masks = {}
    for i_sample, file_name in enumerate(file_list):
        features[file_name] = {}
        print('Computing features for sample ', i_sample+1, '/', len(file_list), ': ', file_name)

        out = output_list[i_sample]
        tgt = target_list[i_sample]

        onset_masks[file_name] = np.any(tgt == 1, axis = 0) #timesteps with note onsets

        features[file_name]['surprise'] = np.sum(np.where(tgt == 1,-np.log2(out), -np.log2(1-out)), axis = 0) #binary cross entropy
        features[file_name]['surprise_positive'] = np.sum(np.where(tgt == 1,-np.log2(out),0), axis = 0)       #positive part of the BCE
        features[file_name]['surprise_negative'] = np.sum(np.where(tgt == 0,-np.log2(1-out),0), axis = 0)     #negative part of the BCE

        features[file_name]['surprise_timing'], features[file_name]['surprise_pitch'] = split_surprise(out, tgt, axis = 0) #when vs. what

        out_normalized = out / np.sum(out, axis = 0)
        features[file_name]['uncertainty'] = joint_entropy(out, axis = 0)                                #uncertainty over note configurations
        (features[file_name]['uncertainty_timing'],
         features[file_name]['uncertainty_pitch_conditional'],
         features[file_name]['uncertainty_pitch_weighted']) = split_uncertainty(out, axis = 0)     #when vs. what
        features[file_name]['uncertainty_legacy'] = np.sum(-np.log2(out_normalized)*out_normalized,axis = 0) #first approximation of uncertainty
        
        features[file_name]['predicted_density'] = np.sum(out, axis = 0)                                 #sum of probabilities
    
    #create & save features summary
    summary = {}
    for i_sample, file_name in enumerate(file_list):
        mask_onset = onset_masks[file_name]
        summary[file_name] = {}
        summary[file_name]['file_name'] = file_name
        summary[file_name]['n_events'] = np.sum(mask_onset)
        summary[file_name]['n_notes'] = target_list[i_sample].sum().astype(int)
        summary[file_name]['duration'] = target_list[i_sample].shape[1] / fs

        for label in features[file_name].keys():
            summary[file_name]['continous_' + label] = np.sum(features[file_name][label])
        for label in features[file_name].keys():
            summary[file_name]['onsets_' + label] = np.sum(features[file_name][label][mask_onset])
    df_summary = pd.DataFrame(summary).T
    df_summary.to_csv(output_path + 'features.csv', sep = ',', index = False)

    #upsample time-resolved features & apply timing correction
    for file_name in file_list:
        features[file_name] = upsample_features(features[file_name], onset_masks[file_name], fs, out_fs,
                                                discrete_features = ['surprise_positive','surprise_pitch'],
                                                onset_times = timings[file_name] if timing_correction else None)

    #save time-resolved features
    if not os.path.exists(output_path + '/features_over_time/'):
        os.makedirs(output_path + '/features_over_time/')
    for file_name in file_list:
        out = np.stack([features[file_name][feature] for feature in features[file_name].keys()],axis=1)
        np.savetxt(output_path + '/features_over_time/' + file_name + '.csv', out, delimiter = ',', header = ','.join(features[file_name].keys()), comments = '')

def export_predictions_activations(data_path, output_path, model_name = 'PolyTNN'):
    '''
    description: export the predictions and hidden layer activations (contextual embeddings) of a trained model for all midi files
        of a folder, in a single pickle file: output_path/predictions_activations_<model_name>.pkl

        Alignment: for each timestep t of the piano roll, outputs[:, t] is the model's prediction about timestep t, knowing the
        timesteps before t, and layers[name][:, t] is the predictive state that produced this prediction (it has integrated the
        piano roll up to t-1, but not timestep t). The state that has integrated timestep t is layers[name][:, t+1].

    parameter: data_path, path to the folder containing the midi files
    parameter: output_path, folder where the pickle file is saved
    parameter: model_name, name of the model folder in ./versions/

    return: dict saved in the pickle file, with one list entry per midi file (same order in all lists):
        - file_names: file names (without extension)
        - samples: model input piano rolls, arrays (88 x time), with the model's onset and sustain values (info.pt)
        - outputs: predicted onset probabilities, arrays (88 x time)
        - layers: dicts {layer name: array (units x time)}
            - transformer: output of each encoder layer ('layer_0', 'layer_1', ...)
            - lstm: hidden state of the last LSTM layer ('layer_<n_lstm_layers-1>')
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model, info = load_model(model_name, device)
    samples, file_names, _ = load_piano_rolls(data_path, info, device)

    results = {'samples': [], 'outputs': [], 'layers': [], 'file_names': file_names}
    for i_sample, (sample, file_name) in enumerate(zip(samples, file_names)):
        print('Computing predictions and activations for sample ', i_sample+1, '/', len(samples), ': ', file_name)
        output, _, layers = predict_piano_roll(model, sample, info, device, return_layers = True)
        results['samples'].append(sample.cpu().numpy().T.astype(np.float32))
        results['outputs'].append(output.cpu().numpy().T.astype(np.float32))
        results['layers'].append({name: act.cpu().numpy().T.astype(np.float32) for name, act in layers.items()})

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f'predictions_activations_{model_name}.pkl'), 'wb') as f:
        pkl.dump(results, f)
    return results
