import librosa as librosa
import torch
import torchaudio


def zero_crossing_rate(audio: torch.Tensor):
    signs = torch.sign(audio)
    zero_crossings = torch.zeros(audio.size(0) - 1, device=audio.device)

    for i in range(audio.size(0) - 1):
        zero_crossings[i] = torch.sum(signs[i] != signs[i + 1])

    zcr = torch.mean(zero_crossings) / (audio.size(0) - 1)
    return zcr.item()


if __name__ == "__main__":
    path = "./parsed_data/classical/test/1.mp3"
    y, sr = torchaudio.load(path, format='mp3')
    print(zero_crossing_rate(y))




