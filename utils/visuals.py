import os
import copy
import numpy as np

import pretty_midi
import matplotlib
import matplotlib.pyplot as plt
import pydub
import cv2
import moviepy.editor as mpe

import torch

from utils.midi_processing import convert_midi_to_piano_roll
from models import Transformer, LSTM
from utils.export import predict_piano_roll, joint_entropy

def visualize_sequence(src, tgt, pred):
    sample = src.detach().cpu().numpy().T
    target = tgt.detach().cpu().numpy().T
    pred = pred.detach().cpu().numpy().T

    fig, axs = plt.subplots(4,1, figsize=(10,10))
    im0 = axs[0].imshow(sample, aspect='auto', interpolation='nearest', vmax=1, vmin=0)
    im1 = axs[1].imshow(target[:88,:], aspect='auto', interpolation='nearest', vmax=1, vmin=0)
    im2 = axs[2].imshow(np.power(pred[:88,:], 0.5), aspect='auto', interpolation='nearest', vmax=1, vmin=0)
    im3 = axs[3].imshow(target[:88,:]-np.power(pred[:88,:], 0.5), aspect='auto', interpolation='nearest', vmax=1, vmin=-1, cmap='bwr')

    #add colorbar to axis
    fig.colorbar(im0, ax=axs[0])
    fig.colorbar(im1, ax=axs[1])
    fig.colorbar(im2, ax=axs[2])
    fig.colorbar(im3, ax=axs[3])

    #add title to axis
    axs[0].title.set_text('Input')
    axs[1].title.set_text('Target')
    axs[2].title.set_text('Prediction')
    axs[3].title.set_text('Difference')

    fig.tight_layout()
    return fig

def make_video(file_path = './data/midi_dataset_example/test/',
               file_name = 'MIDI-Unprocessed_059_PIANO059_MID--AUDIO-split_07-07-17_Piano-e_2-03_wav--1.mid',
               res = (1280,720), 
               graph = True,
               model_name = 'transformer_maestro_test',
               ffmpeg_path = r'C:/ffmpeg/'):
    '''
    description: Create of video (.mp4) of a model's output on a midi file.
    For this function to work, ffmpeg must be installed on the computer (see: https://www.ffmpeg.org/download.html).
    The video will be saved in ./versions/model_name/video/.
    
    parameter: file_path, the path to the midi file
    parameter: file_name, the name of the midi file
    parameter: res, the resolution of the video (width, height)
    parameter: graph, if True, display the surprise and uncertainty graphs
    parameter: model_name, the name of the model folder in ./versions/
    parameter: ffmpeg_path, the path to ffmpeg 
    '''
    
    #set graph limits
    if graph:
        surprise_graph_lims = [0,25]
        uncertainty_graph_lims = [0,16] #bits (theoretical maximum: 88 bits, but values above ~15 bits are rare in practice)

    #set brightness of model's output
    brightness = 0.5        #model outputs (in range [0;1]) are raised to the power of (1 - brightness) to increase visual contrast
                            #set the brightness value between 0 and 1.
    
    #set matplotlib backend
    original_backend = matplotlib.get_backend()
    matplotlib.use('TkAgg')
    
    #set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    #create tmp folder
    os.makedirs('./tmp_video', exist_ok=True)

    #model import
    model = torch.load('./versions/'+ model_name +'/model.pt', weights_only=False, map_location=device) #model class must be imported
    model.device = device
    info = torch.load('./versions/' + model_name + '/info.pt', map_location=device)
    fs = info['fs']
    ons_value = info['ons_value']
    sus_value = info['sus_value']

    #convert midi to piano roll
    convert_midi_to_piano_roll(data_path = file_path, out_dir = './tmp_video/', file_name = file_name,
                               fs = fs, pedal_threshold = 64)
    sample_name = file_name.split('.')[0] + '.pt'
    pr = torch.load('./tmp_video/' + sample_name, map_location=device)
    sample = torch.zeros(pr.shape)
    sample[pr == 1] = sus_value
    sample[pr == 2] = ons_value

    #get model predictions (output[t] is the prediction about sample[t], knowing sample[:t])
    output, tgt = predict_piano_roll(model, sample, info, device)

    #compute features to plot
    if graph:
        p = output.cpu().numpy()
        t = tgt.cpu().numpy()

        # overall suprise (binary cross entropy)
        # surprise = np.where(t == 1, -np.log2(p), -np.log2(1-p))
        # surprise = np.sum(surprise, axis = 1)

        # maximum positive suprise among simultaneous notes
        surprise = np.max(-np.log2(p) * t, axis = 1)
        surprise[np.isnan(surprise)] = 0

        #uncertainty
        uncertainty = joint_entropy(p, axis = 1)


    #reformat model input & output for display
    output = output.cpu().numpy()
    sample = sample.cpu().numpy()

    #set model_input / model_output / features_graph screen ratios
    res_input = (5*fs,88)
    if graph:
        io_ratio = (0.5,0.2,0.3)
    else:
        io_ratio = (0.8,0.2,0)

    #set figure
    if graph:
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig = plt.figure(figsize = (int(np.ceil(res[0]*io_ratio[2]))*px,res[1]*px), dpi=100)
        gs = fig.add_gridspec(4, 1, height_ratios = [0.5, 1, 1, 0.5]) #graphs in the middle rows, so that each keeps ~1/3 of the height
        axs = [fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
        x_graph = np.arange(res_input[0])
        axs[0].title.set_text('Surprise')
        axs[0].set_ylim(surprise_graph_lims)
        axs[0].set_xlim([0,len(x_graph)])
        axs[0].get_xaxis().set_visible(False)

        axs[1].title.set_text('Uncertainty')
        axs[1].set_ylim(uncertainty_graph_lims)
        axs[1].set_xlim([0,len(x_graph)])
        axs[1].get_xaxis().set_visible(False)

        fig.tight_layout()
    
    #define video object
    fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    writer = cv2.VideoWriter('./tmp_video/__temp__' + '.avi', fourcc, fps = float(fs), frameSize = res)

    #write video
    for i_step in range(0, sample.shape[0] - 1 +1):
        if i_step < res_input[0]:
            input = sample[:i_step,:]
            if graph:
                empty_array = np.full([res_input[0]-i_step], np.nan)
                surprise_ = np.concatenate((empty_array,surprise[:i_step]))
                uncertainty_ = np.concatenate((empty_array,uncertainty[:i_step]))
        else:
            input = sample[i_step-res_input[0]:i_step,:]
            if graph:
                surprise_ = surprise[i_step-res_input[0]:i_step]
                uncertainty_ = uncertainty[i_step-res_input[0]:i_step]
        out = output[i_step,:]
        output_display = copy.copy(out.reshape(1,88))
        output_display = np.power(output_display, 1 - brightness)

        if input.shape[0] >= res_input[0]:
                input_display = copy.copy(input[-res_input[0]:,:])
        else:
            empty_array = np.zeros([res_input[0]-input.shape[0], 88])
            input_display = np.concatenate((empty_array,copy.copy(input)),axis=0)
        input_display /= 2 

        #plot
        if graph:
            _s = axs[0].plot(x_graph,surprise_,color='black')
            _p = axs[1].plot(x_graph,uncertainty_,color='black')
        #convert to cv2 image
        if graph:
            fig.canvas.draw()
            plot_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                    sep='')
            plot_img  = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plot_img = cv2.cvtColor(plot_img,cv2.COLOR_RGB2BGR)
            _s.pop(0).remove()
            _p.pop(0).remove()

        ## visual output
        input_img = cv2.resize(input_display, dsize=(res[1],np.floor(res[0]*io_ratio[0]).astype(int)), interpolation=cv2.INTER_AREA)
        output_img = cv2.resize(output_display, dsize=(res[1],np.ceil(res[0]*io_ratio[1]).astype(int)), interpolation=cv2.INTER_AREA)
        img = np.concatenate((input_img,output_img),axis=0)                         # concatenate input & output
        img[int(img.shape[0]*(io_ratio[0]/(io_ratio[0]+io_ratio[1]))),:] = 1        # draw vertical white line
        img = (np.stack([img]*3, axis=2) * 255).astype(int)                         # to RGB
        if graph:
            plot_img_res = (int(np.ceil(res[0]*io_ratio[2])),int(res[1]))
            plot_img = cv2.resize(plot_img, dsize=plot_img_res, interpolation=cv2.INTER_AREA)
            img = np.concatenate((img,np.transpose(plot_img,(1,0,2))),axis=0)
        img = img.astype('uint8')
        img = np.transpose(img, (1,0,2))

        if i_step%100 == 0 or i_step == sample.shape[0] - 1:
            print(i_step, '/', sample.shape[0]-1)
        writer.write(img)   

    writer.release()
    cv2.destroyAllWindows()
    plt.close()

    #set audio library
    pydub.AudioSegment.converter = ffmpeg_path + r'\bin\ffmpeg.exe'
    pydub.AudioSegment.ffprobe   = ffmpeg_path + r'\bin\ffprobe.exe'

    #generate audio
    score = pretty_midi.PrettyMIDI(file_path+file_name)
    wv = score.fluidsynth(fs=44100, sf2_path='./utils/GU_font.sf2')
    y = np.int16(wv * 2 ** 15)
    audio = pydub.AudioSegment(y.tobytes(), 
                                sample_width=2, 
                                frame_rate=44100,
                                channels=1)
    audio.export('./tmp_video/__temp__' + '.mp3', format='mp3')

    ## add audio to video
    videoclip = mpe.VideoFileClip('./tmp_video/__temp__.avi')
    audioclip = mpe.AudioFileClip('./tmp_video/__temp__.mp3')
    new_audioclip = mpe.CompositeAudioClip([audioclip])
    videoclip.audio = new_audioclip

    #write final video
    if not os.path.exists('./versions/' + model_name + '/video/'):
        os.makedirs('./versions/' + model_name + '/video/')
    out_name = file_name.split('.')[0] + '.mp4'
    videoclip.write_videofile('./versions/' + model_name + '/video/' + out_name,codec= 'libx264')
    videoclip.close()
    audioclip.close()

    #clean tmp folder with its content
    for file in os.listdir('./tmp_video/'):
        os.remove('./tmp_video/' + file)
    os.rmdir('./tmp_video/')

    #reset matplotlib backend
    matplotlib.use(original_backend)