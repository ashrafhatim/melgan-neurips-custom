import scipy.io.wavfile

import torch
import torch.nn.functional as F

import statistics



def save_sample(file_path, sampling_rate, audio):
    """Helper function to save sample

    Args:
        file_path (str or pathlib.Path): save file path
        sampling_rate (int): sampling rate of audio (usually 22050)
        audio (torch.FloatTensor): torch array containing audio in [-1, 1]
    """
    audio = (audio.numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)


def mel_rec_val_loss(val_loader, netG, fft):
    errors = []
    with torch.no_grad():
        for x_t in val_loader:
            x_t = x_t.cuda()
            s_t = fft(x_t)
            x_pred_t = netG(s_t.cuda())
            s_pred_t = fft(x_pred_t)
            
            s_error = F.l1_loss(s_t, s_pred_t).item()
            errors.append(s_error)
            
    return statistics.mean(errors)

       