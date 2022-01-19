# Unofficial repository for the paper MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis
## This code is intended to tackle the Arabic Text-to-speech task considering it as low-resource problem

Previous works have found that generating coherent raw audio waveforms with GANs is challenging. In this [paper](https://arxiv.org/abs/1910.06711), they show that it is possible to train GANs reliably to generate high quality coherent waveforms by introducing a set of architectural changes and simple training techniques. Subjective evaluation metric (Mean Opinion Score, or MOS) shows the effectiveness of the proposed approach for high quality mel-spectrogram inversion. The model is non-autoregressive, fully convolutional, with significantly fewer parameters than competing models and generalizes to unseen speakers for mel-spectrogram inversion. Here we are focusing on training the Arabic TTS system with adversarial networks using publicly available data, evaluating the efficiency of MelGANs for low-resource speech datasets. The code is inspired by the official representation of [MelGAN](https://github.com/descriptinc/melgan-neurips). 



## Code organization

    ├── README.md             <- Top-level README.
    ├── set_env.sh            <- Set PYTHONPATH and CUDA_VISIBLE_DEVICES.
    │
    ├── mel2wav
    │   ├── dataset.py           <- data loader scripts
    │   ├── modules.py           <- Model, layers and losses
    │   ├── utils.py             <- Utilities to monitor, save, log, schedule etc.
    │   ├── custom_transforms.py <- data transformations.
    │
    ├── scripts
    │   ├── train.py                    <- training / validation / etc scripts
    │   ├── generate_from_folder.py


## Preparing dataset
Create a raw folder with all the samples stored in `wavs/` subfolder.
Run these commands:
   ```command
   ls wavs/*.wav | tail -n+10 > train_files.txt
   ls wavs/*.wav | head -n10 > test_files.txt
   ```

## Training Example
    . source set_env.sh 0
    # Set PYTHONPATH and use first GPU
    python scripts/train.py --save_path logs/baseline --path <root_data_folder>

