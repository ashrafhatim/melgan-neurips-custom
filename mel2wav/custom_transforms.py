# from librosa.core import load
# from librosa.util import normalize
# import numpy as np
# import torch

import librosa
import torch
import random

class change_speed(object):
    """Change the speed of the audio.

    Args:
        speed_facor (list or int): Desired speed of the resulted audio after the transformation. Or the range of the desired speed [start, end].
        step: the steps between the desired speed range.
    """

    def __init__(self, speed_facor, step=0.001):
        self.step = step
        if isinstance(self.speed_factor, list):
            self.speed_facor = range(self.speed_facor[0], self.speed_facor[1], self.step)
        else:
            self.speed_facor = speed_facor

    def __call__(self, sample):
        # sample = sample.numpy()

        if isinstance(self.speed_factor, range):
            factor = random.choice(self.speed_facor) 
        else:
            factor = self.speed_factor

        output = librosa.effects.time_stretch(sample, factor)

        return output