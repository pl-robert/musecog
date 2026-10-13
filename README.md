# MuseCog

**MuseCog** is a repository of neural network models designed to simulate cognitive processes underlying music perception. These models are intended to support research on perceptual and affective mechanisms in behavioral and neuroimaging studies. Current implementations focus on modeling listener expectations for upcoming notes and rhythms in polyphonic music.

<div align="center">
  <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExamx0dnI3enBsMXJ1eHFubzhsajlpcTkxd2wxcDJ2eGV0Yjc5dGZ6cSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/lZOf8i68nhLbE5WfMk/giphy.webp" alt="PolyRNN Demo">
  <br>
  <em>PolyRNN predictions in Bach's prelude in C# Major</em>
  <br>
  <em>- <a href="https://www.youtube.com/watch?v=WTHKQMljzXY">demo with sound</a> -</em>
  <br><br>
</div>

This repository contains:
* PolyRNN: an LSTM model built to yield time-resolved predictions in polyphonic music.
* PolyTNN: equivalent to PolyRNN, with a transformer architecture.

More information about this framework can be found in Robert et al. (2024),
please use this as citation in publications using this software.

Robert et al., (2024). [PolyRNN: A time-resolved model of polyphonic musical expectations aligned with human brain responses](https://www.biorxiv.org/content/10.1101/2024.11.27.625704v2). *bioRxiv, 2024-11*

## Setup

In this initial release, the code has been tested on Windows 11 with [Python 3.9.13](https://www.python.org/downloads/release/python-3913/).
Estimated installation time: ~ 40 minutes.

### CUDA install

GPU acceleration is necessary to train transformer models, and highly recommanded to
train neural networks in general.
To benefit from GPU acceleration, make sure you have a GPU compatible with CUDA,
and install [CUDA](https://developer.nvidia.com/cuda-downloads).

### Python environment
With Windows Command Prompt:

* clone repository

```bash
git clone https://github.com/pl-robert/musecog
cd musecog
```

* set up a virtual environment

```bash
pip install virtualenv
virtualenv venv
.\venv\Scripts\activate.bat
```

* install python dependencies

```bash
pip install -r requirements.txt
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### Other dependencies
The ```make_video()``` function allows you to make a video of the output of a model for a MIDI file.
Using this function requires to install:
* [fluidsynth](https://github.com/FluidSynth/fluidsynth/releases)
* [ffmpeg](https://www.ffmpeg.org/download.html)

Note that the path of the ffmpeg folder will have to be given as argument to the function ```make_video()```. Fluidsynth will be automatically located,
and can be installed anywhere.

## Usage
The ```./train_lstm.ipynb``` and ```./train_transformer.ipynb``` notebooks contain the code to
preprocess a MIDI dataset, train a recurrent or transformer model, visualize the model's output,
and export a set of features from your MIDI stimuli. You can open these files with the command ```jupyter notebook train_lstm.ipynb```.

An small midi dataset is provided in ```./data/midi_dataset_example/```. As neural networks require a large amount of data for
training, we recommand using pre-built MIDI datasets like [MAESTRO V3.0.0](https://magenta.tensorflow.org/datasets/maestro).

Pre-trained models PolyRNN and PolyTNN are provided in ```./versions/```, and can be accessed directly with the ```make_video()```
and ```export_features()``` functions.

Training a new model can take from several hours to several days depending on the training parameters and the hardware.
Generating new predictions with a trained model generally takes a few minutes, depending on the size of the dataset (GPU is required for transformer models).









