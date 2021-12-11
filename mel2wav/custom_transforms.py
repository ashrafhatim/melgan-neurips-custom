import librosa
import random
import numpy as np

class change_speed(object):
    """Change the speed of the audio.

    Args:
        speed_facor (list or int): Desired speed of the resulted audio after the transformation. Or the range of the desired speed [start, end].
        step: the steps between the desired speed range.
    """

    def __init__(self, speed_factor, step=0.001):
        self.step = step
        if isinstance(speed_factor, list):
            self.speed_factor = list(np.arange(speed_factor[0], speed_factor[1], self.step))
        else:
            self.speed_factor = speed_factor

    def __call__(self, sample):
        # sample = sample.numpy()
        if isinstance(self.speed_factor, list):
            factor = random.choice(self.speed_factor) 
        else:
            factor = self.speed_factor

        output = librosa.effects.time_stretch(sample, factor)

        return output