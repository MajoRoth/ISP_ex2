import torch
import torchaudio
import librosa
import numpy as np


def zero_crossing_rate(audio: torch.Tensor):
    signs = torch.sign(audio)
    zero_crossings = signs[:, :-1] != signs[:, 1:]
    zcr = zero_crossings.float().mean(dim=-1)
    return zcr


def extract_rms(wavs: torch.Tensor):
    FRAME_SIZE = 2048
    HOP_LENGTH = 1024
    rms = librosa.feature.rms(y=wavs[None, :, :], frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    return torch.from_numpy(rms[:, 0, :])

if __name__ == "__main__":
    path = "./parsed_data/classical/test/1.mp3"
    y, sr = torchaudio.load(path, format='mp3')
    print(zero_crossing_rate(y))




