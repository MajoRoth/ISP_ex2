import torch
import librosa


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


def extract_mfccs(wavs: torch.Tensor, sr: int):
    FRAME_SIZE = 2048
    HOP_LENGTH = 1024
    mfccs = librosa.feature.mfcc(y=wavs.numpy(), sr=sr, win_length=FRAME_SIZE, hop_length=HOP_LENGTH)
    return torch.from_numpy(mfccs.reshape((mfccs.shape[0], -1)))





