import torch
import torch.utils.data
import torch.nn.functional as F

from librosa.core import load
from librosa.util import normalize

from pathlib import Path
import numpy as np
import random


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, training_files, segment_length, sampling_rate, augment=True, transform=None, training_files_english=None):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.audio_files = files_to_list(training_files)
        self.audio_files = [Path(training_files).parent / x for x in self.audio_files]
        random.seed(1234)
        random.shuffle(self.audio_files)

        if training_files_english:
            self.audio_files_english = files_to_list(training_files_english)
            self.audio_files_english = [Path(training_files_english).parent / x for x in self.audio_files_english]
            random.shuffle(self.audio_files_english)
            self.num_english_files = len(self.audio_files_english)

        self.augment = augment
        self.transform=transform
        

    def __getitem__(self, index):
        # Read audio
        
        # # Take segment
        # if audio.size(0) >= self.segment_length:
        #     max_audio_start = audio.size(0) - self.segment_length
        #     audio_start = random.randint(0, max_audio_start)
        #     audio1 = audio[audio_start : audio_start + self.segment_length]
        # else:
        #     audio1 = F.pad(
        #         audio, (0, self.segment_length - audio.size(0)), "constant"
        #     ).data

        # Take segment1
        out = []
        
        # main discs
        filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(filename)

        if audio.size(0) >= self.segment_length[0]:
            max_audio_start = audio.size(0) - self.segment_length[0]
            audio_start = random.randint(0, max_audio_start)
            out.append( audio[audio_start : audio_start + self.segment_length[0]].unsqueeze(0) )
        else:
            out.append( F.pad(
                audio, (0, self.segment_length[0] - audio.size(0)), "constant"
            ).data.unsqueeze(0) )


        # helper disc
        if len(self.segment_length) > 1:

            # options = ["arabic", "english"]
            # lang = random.choices(options, weights=[1,1], k=1)[0]

            # if lang=="english":

            index = random.randint(0, self.num_english_files-1)
            filename = self.audio_files_english[index]
            audio, sampling_rate = self.load_wav_to_torch(filename)
                
            if audio.size(0) >= self.segment_length[1]:
                max_audio_start = audio.size(0) - self.segment_length[1]
                audio_start = random.randint(0, max_audio_start)
                # segment1 = audio[audio_start : audio_start + self.segment_length[1]].unsqueeze(0)

                # audio_start = random.randint(0, max_audio_start)
                # segment2 = audio[audio_start : audio_start + self.segment_length[1]].unsqueeze(0)

                # out.append( torch.concat((segment1,segment2), axis=0) )
                out.append( audio[audio_start : audio_start + self.segment_length[1]].unsqueeze(0))
            else:
                out.append( F.pad(
                    audio, (0, self.segment_length[1] - audio.size(0)), "constant"
                ).data.unsqueeze(0) )
                
                

            # elif lang=="english":
            #     index = random.choice(list(range(len(self.audio_files_english))))
            #     filename = self.audio_files_english[index]
            #     audio, sampling_rate = self.load_wav_to_torch(filename)

            #     if audio.size(0) >= self.segment_length[1]:
            #         max_audio_start = audio.size(0) - self.segment_length[1]
            #         audio_start = random.randint(0, max_audio_start)
            #         out.append( audio[audio_start : audio_start + self.segment_length[1]].unsqueeze(0) )
            #     else:
            #         out.append( F.pad(
            #             audio, (0, self.segment_length[1] - audio.size(0)), "constant"
            #         ).data.unsqueeze(0) )

        


        # audio = audio / 32768.0
        # return audio.unsqueeze(0)
        
        # make sure that val_loader does not receive list
        if len(out) == 1:
            return out[0]

        return out

    def __len__(self):
        return len(self.audio_files)

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=self.sampling_rate)
        data = 0.95 * normalize(data)

        # amplitude = np.random.uniform(low=0.3, high=1.0)
        # data = data * amplitude

        if self.augment:
            # amplitude = np.random.uniform(low=0.3, high=1.0)
            # data = data * amplitude
             
            data = self.transform(data)

        return torch.from_numpy(data).float(), sampling_rate
