import os
import pickle as pkl
import numpy as np
import torch
import pretty_midi as pm

def convert_midi_to_piano_roll(data_path = './data/', out_dir = './data/piano_rolls/', file_name = None,
                               fs = 30, pedal_threshold = 64,
                               save_midi_timings = False):
    '''
    description: converts midi files to piano rolls and saves them as pytorch tensors

    parameter: data_path, path to the midi files
    parameter: out_dir, path to the directory where the piano rolls will be saved
    parameter: file_name, name of the midi file to convert. If None, all midi files in the data_path will be converted.
    parameter: fs, sampling frequency
    parameter: pedal_threshold, threshold (0-127) to trigger the sustain pedal
    parameter: save_midi_timings, if True, the exact timing of note onsets will be saved in a pickle file (used for timing correction when exporting features)
    
    returns:
    '''

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    files = os.listdir(data_path)

    # Assign a value to the onset and sustain of notes. This value can be changed later, 
    # when loading the data with the dataloader (PianoRollDataset), to adjust the model's input.
    # Until then, the onset and sustain values are fixed (do not change it !).
    ons_value = 2                   
    sus_value = 1
    
    #convert midi files to piano rolls and save them as pytorch tensors
    if save_midi_timings:
        timings = {} #keep track of the exact timing of note onsets from midi files
    for idx in range(len(files)):
        file_name = files[idx]
        if file_name[-4:] == '.mid' or file_name[-4:] == '.MID' or file_name[-5:] == '.midi' or file_name[-5:] == '.MIDI':
            print('Preprocessing midi file: ', file_name)

            pm_object = pm.PrettyMIDI(data_path + file_name)
            pr, onsets_timing, _ = onset_piano_roll(pm_object, fs=fs, 
                                        onset_value = ons_value, sustain_value = sus_value,
                                        pedal_threshold=pedal_threshold)
            pr = pr[20:108,:]                       #remove notes that cannot be played by a piano
            pr = np.flip(pr,axis=0).copy()          #high pitch first
            pr = torch.tensor(pr, dtype = torch.float32).T
            torch.save(pr, out_dir + file_name.split('.')[0] + '.pt')

            if save_midi_timings:
                timings[file_name.split('.')[0]] = onsets_timing #same name as the saved piano roll

    if save_midi_timings:
        with open(out_dir + 'timings.pkl', 'wb') as f:
            pkl.dump(timings, f)

def onset_piano_roll(pm_object, fs=30, onset_value = 2, sustain_value = 1, pedal_threshold=64):
    '''
    description: converts a pretty_midi object to a piano roll


    parameter: pm_object, pretty_midi object
    parameter: fs, sampling frequency
    parameter: onset_value, value of the onset in the piano roll
    parameter: sustain_value, value of the sustain in the piano roll
    parameter: pedal_threshold, threshold (0-127) to trigger the sustain pedal

    returns: piano_roll, piano roll of the midi file
    returns: all_onsets_timings, list of all onsets timings
    returns: all_time_axis, global time axis
    '''
    
    # If there are no instruments, return an empty array
    if len(pm_object.instruments) == 0:
        return np.zeros((128, 0))
    
    #retrieve piano rolls for each instrument
    piano_rolls = []
    onset_timings = []
    time_axis = []
    for instrument in pm_object.instruments:
        if instrument.is_drum or 97 <= instrument.program <= 104 or 113 <= instrument.program <= 128: #remove drums and effects
            print('Midi Drums / Effects detected, skipping...')
            continue

        pr, ot, t = onset_inst_piano_roll(instrument, fs=fs, 
                        onset_value = onset_value, sustain_value = sustain_value,
                        pedal_threshold=pedal_threshold)
        piano_rolls.append(pr)
        onset_timings.append(ot)
        time_axis.append(t)
    
    # Build final piano roll
    piano_roll = np.zeros((128, np.max([p.shape[1] for p in piano_rolls])))
    for roll in piano_rolls:
        piano_roll[:, :roll.shape[1]] = np.maximum(piano_roll[:, :roll.shape[1]], roll)
        
    #retrieve list of all onsets from all instruments in the midi files
    onset_timings = [ot for inst_ot in onset_timings for ot in inst_ot]
    
    #retrieve global time axis
    time_axis = max(time_axis, key=len)
    
    return piano_roll, onset_timings, time_axis


#custom script to convert a pretty_midi instrument object to a piano roll, derived from the pretty_midi library
def onset_inst_piano_roll(instrument, fs=30, onset_value = 2, sustain_value = 1, pedal_threshold=64):
    '''
    description: converts a pretty_midi instrument object to a piano roll
    parameter: instrument, pretty_midi instrument object
    parameter: fs, sampling frequency
    parameter: onset_value, value of the onset in the piano roll
    parameter: sustain_value, value of the sustain in the piano roll
    parameter: pedal_threshold, threshold (0-127) to trigger the sustain pedal

    returns: piano_roll, piano roll of the midi file
    returns: onset_timings, list of all onsets timings
    returns: time_axis, time axis
    '''
    
    #store onsets timings for timing correction, later
    onset_timings = []
    
    # If there are no notes, return an empty matrix
    if instrument.notes == []:
        return np.array([[]]*128)
    
    # Get the end time of the last event
    end_time = instrument.get_end_time() + 1/fs #(to avoid some precision error in time-axis generation)
    # Allocate a matrix of zeros - we will add in as we go
    piano_roll = np.zeros((128, int(fs*end_time)))
    # Add up piano roll matrix, note-by-note
    for note in instrument.notes:
        if round(note.start*fs) < piano_roll.shape[1]:
          onset_timings.append(note.start)
          for iT in range(round(note.start*fs)+1,int(note.end*fs)):
            if piano_roll[note.pitch,iT] != onset_value:
              piano_roll[note.pitch,iT] = sustain_value
          piano_roll[note.pitch,
                    round(note.start*fs)] = onset_value
    
    # Process sustain pedals
    if pedal_threshold is not None:
        cc_sustain_pedal = 64
        time_pedal_on = 0
        is_pedal_on = False
        for cc in [_e for _e in instrument.control_changes
                   if _e.number == cc_sustain_pedal]:
            time_now = int(cc.time*fs)
            is_current_pedal_on = (cc.value >= pedal_threshold)
            if not is_pedal_on and is_current_pedal_on:
                time_pedal_on = time_now
                is_pedal_on = True
            elif is_pedal_on and not is_current_pedal_on:
                # For each pitch, a sustain pedal "retains"
                # the maximum velocity up to now due to
                # logarithmic nature of human loudness perception
                subpr = piano_roll[:, time_pedal_on:time_now]
    
                # # Take the running maximum
                # pedaled = np.maximum.accumulate(subpr, axis=1)
                pedaled = subpr.copy()
                for iRow in range(pedaled.shape[0]):
                  note_on = False
                  for iCol in range (pedaled.shape[1]):
                    if note_on == True and pedaled[iRow,iCol] != onset_value and onset_value is not None:
                      pedaled[iRow,iCol] = sustain_value
                    if pedaled[iRow,iCol] > 0:
                      note_on = True
                piano_roll[:, time_pedal_on:time_now] = pedaled
                is_pedal_on = False
    
    onset_timings = np.array(onset_timings)
    time_axis = np.linspace(0.0, (1/fs)*(piano_roll.shape[1]-1), num=piano_roll.shape[1])
    
    return piano_roll, onset_timings, time_axis