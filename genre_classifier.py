from abc import abstractmethod

import librosa.feature
import numpy as np
from enum import Enum
import typing as tp
from dataclasses import dataclass

import torch
from tqdm import tqdm


from torch.utils.data import Dataset
import json
import torchaudio

class Genre(Enum):
    """
    This enum class is optional and defined for your convinience, you are not required to use it.
    Please use the int labels this enum defines for the corresponding genras in your predictions.
    """
    CLASSICAL: int=0
    HEAVY_ROCK: int=1
    REGGAE: int=2


label2genre = {"classical": Genre.CLASSICAL,
               "heavy-rock": Genre.HEAVY_ROCK,
               "reggae": Genre.REGGAE}


class Mp3Dataset(Dataset):
    def __init__(self, files_json):
        self.files = json.load(open(files_json))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        y, sr = torchaudio.load(self.files[index]['path'])
        label = label2genre[self.files[index]['label']].value
        return {"waveform": y, "label": label}


@dataclass
class TrainingParameters:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with 
    default values (so run won't break when we test this).
    """
    batch_size: int = 32
    num_epochs: int = 20
    train_json_path: str = "jsons/train.json" # you should use this file path to load your train data
    test_json_path: str = "jsons/test.json" # you should use this file path to load your test data
    # other training hyper parameters


@dataclass
class OptimizationParameters:
    """
    This dataclass defines optimization related hyper-parameters to be passed to the model.
    feel free to add/change it as you see fit.
    """
    learning_rate: float = 0.001


class MusicClassifier:
    """
    You should Implement your classifier object here
    """

    def __init__(self, opt_params: OptimizationParameters = None, encoding_dim=40, **kwargs):
        """
        This defines the classifier object.
        - You should define your weights and biases as class components here.
        - You could use kwargs (dictionary) for any other variables you wish to pass in here.
        - You should use `opt_params` for your optimization and you are welcome to experiment
        """
        self.opt_params = opt_params
        self.encoding_dim = encoding_dim  # input size
        self.num_classes = len(label2genre)  # output size

        INIT_STD = 0.01
        self.W = torch.randn((self.encoding_dim, self.num_classes)) * INIT_STD
        self.b = torch.zeros((1, self.num_classes))

        self.loss = list()
        self.data_sr = 22050

        self.features_means = torch.zeros((1, self.encoding_dim)).float()
        self.features_stds = torch.ones((1, self.encoding_dim)).float()

    def estimate_feats_distribution(self, train_dataloader):
        all_train_feats = None
        NUM_BATCHS_TO_USE = 10
        for i, raw_features in enumerate(train_dataloader):
            feats = self.exctract_feats(raw_features['waveform'])
            if all_train_feats is None:
                all_train_feats = feats
            else:
                all_train_feats = torch.concat((all_train_feats, feats), dim=0)
            if i > NUM_BATCHS_TO_USE:
                break
            print(f"estimating features distribution - i: {i}/{NUM_BATCHS_TO_USE}")

        self.features_means = all_train_feats.mean(dim=0)
        self.features_stds = all_train_feats.std(dim=0)


    def normalized_features(self, features):
        return (features - self.features_means) / self.features_stds


    def exctract_feats(self, wavs: torch.Tensor):
        """
        this function extract features from a given audio.
        we will not be observing this method.

        Return: [batch, encoding_dim]
        """

        WIN_LEN = 2048
        HOP_LEN = 1024

        b, audio_channels, n = wavs.shape
        assert audio_channels == 1 and "assuming mono"
        wavs = wavs[:, 0, :]

        # feature extraction
        wavs_numpy = wavs.numpy()

        # zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=wavs_numpy,frame_length=WIN_LEN, hop_length=HOP_LEN)[:, 0, :]
        zcr = np.concatenate((np.mean(zcr, axis=-1, keepdims=True), np.std(zcr, axis=-1, keepdims=True)), axis=-1)

        # mfcc
        mfccs = librosa.feature.mfcc(y=wavs_numpy, sr=self.data_sr,
                                     n_fft=WIN_LEN, hop_length=HOP_LEN)
        mfccs = np.mean(mfccs, axis=-1).reshape((b, -1))

        # chroma
        chroma_stft = librosa.feature.chroma_stft(y=wavs_numpy, sr=self.data_sr, n_chroma=12,
                                                  n_fft=WIN_LEN, hop_length=HOP_LEN)
        chroma_stft = np.mean(chroma_stft, axis=-1).reshape((b, -1))

        # spec centroid
        spec_centroid = librosa.feature.spectral_centroid(y=wavs_numpy, sr=self.data_sr,
                                                          n_fft=WIN_LEN, hop_length=HOP_LEN)[:, 0, :]
        spec_centroid = \
            np.concatenate((np.mean(spec_centroid, axis=-1, keepdims=True),
                            np.std(spec_centroid, axis=-1, keepdims=True)), axis=-1)

        # rms
        rms = librosa.feature.rms(y=wavs[None, :, :],
                                  frame_length=WIN_LEN, hop_length=HOP_LEN)[0, :, 0, :]

        # rms derivative
        d_rms = rms[:, 1:] - rms[:, :-1]

        rms = np.concatenate((np.mean(rms, axis=-1, keepdims=True), np.std(rms, axis=-1, keepdims=True)), axis=-1)
        d_rms = np.concatenate((np.mean(d_rms, axis=-1, keepdims=True), np.std(d_rms, axis=-1, keepdims=True)), axis=-1)

        # concat all features
        feats = np.concatenate((zcr, mfccs, chroma_stft, spec_centroid, rms, d_rms), axis=-1)
        feats = torch.from_numpy(feats).float()

        # normalize
        return self.normalized_features(feats)

    def forward(self, feats: torch.Tensor) -> tp.Any:
        """
        this function performs a forward pass throuh the model, outputting scores for every class.
        feats: batch of extracted faetures
        """
        # feats [b, d]
        # W [d, 3]
        Z = torch.matmul(feats, self.W) + self.b
        return torch.softmax(Z, dim=-1)

    def backward(self, feats: torch.Tensor, output_scores: torch.Tensor, labels: torch.Tensor):
        """
        this function should perform a backward pass through the model.
        - calculate loss
        - calculate gradients
        - update gradients using SGD

        Note: in practice - the optimization process is usually external to the model.
        We thought it may result in less coding needed if you are to apply it here, hence 
        OptimizationParameters are passed to the initialization function
        """

        CLASS_IMPORTANCE = torch.tensor([1.0, 1.0, 10.0]).unsqueeze(0) # boosting Raggae genre on which learning was slower

        gt_scores = torch.nn.functional.one_hot(labels)

        # grad calculation
        error = (gt_scores - output_scores) * CLASS_IMPORTANCE
        batch = feats.shape[0]
        feats = torch.concat((feats, torch.ones((batch, 1))), dim=-1)  # adding ones for bias term
        self.grad = (1 / float(batch)) * -torch.matmul(error.T, feats)  # error * x  is the analytic gradient

        REGULARIZATION = True
        if REGULARIZATION:
            MU = 0.01
            self.grad += 2 * MU * torch.concat((self.W, self.b), dim=0).T

        # update W and b
        self.W -= self.opt_params.learning_rate * self.grad[:, :-1].T
        self.b -= self.opt_params.learning_rate * self.grad[:, -1].unsqueeze(0)

    def train_step(self, raw_features):
        # forward
        features = self.exctract_feats(raw_features['waveform'])

        y_preds = self.forward(features)

        # loss and accuracy calculation for evaluation purposes
        labels_onehot = torch.nn.functional.one_hot(raw_features['label'], num_classes=len(label2genre))
        loss = -torch.log((y_preds * labels_onehot).sum(dim=-1)).mean()
        accuracy = (torch.argmax(y_preds, dim=-1) == raw_features['label']).float().mean()

        # backward
        self.backward(features, output_scores=y_preds, labels=raw_features['label'])

        return loss, accuracy

    def validation_forward(self, raw_features):
        # forward
        features = self.exctract_feats(raw_features['waveform'])
        y_preds = self.forward(features)

        n_classes = len(label2genre)

        # loss calculation
        labels_onehot = torch.nn.functional.one_hot(raw_features['label'], num_classes=n_classes)
        loss = -torch.log((y_preds * labels_onehot).sum(dim=-1)).mean()

        # accuracies
        hits = torch.argmax(y_preds, dim=-1) == raw_features['label'] # for accuracy calculation

        genre_hits = [None for _ in range(n_classes)] # for the per genre accuracy
        for c in range(n_classes):
            genre_hits[c] = hits[raw_features['label'] == c]

        return loss, hits, genre_hits

    def get_weights_and_biases(self):
        """
        This function returns the weights and biases associated with this model object, 
        should return a tuple: (weights, biases)
        """
        return self.W, self.b

    def classify(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        this method should recieve a torch.Tensor of shape [batch, channels, time] (float tensor) 
        and a output batch of corresponding labels [B, 1] (integer tensor)
        """
        return torch.argmax(self.forward(self.exctract_feats(wavs)), dim=-1)

    def save(self, name: str):
        m = {'W': self.W,
             'b': self.b,
             'feats_means': self.features_means,
             'feats_stds': self.features_stds,
             'opt': OptimizationParameters}
        torch.save(m, f"model_files/{name}.pt")

    def load(self, name: str):
        loaded = torch.load(f"model_files/{name}.pt")
        self.W = loaded["W"]
        self.b = loaded["b"]
        self.features_means = loaded["feats_means"]
        self.features_stds = loaded["feats_stds"]
        self.opt_params = loaded['opt']

    @staticmethod
    def softmax(input):
        shifted_input = input - np.max(input)
        exp_input = np.exp(shifted_input)
        softmax_probs = exp_input / np.sum(exp_input)
        return softmax_probs

    @staticmethod
    def cross_entropy(y, probs):
        return -1 * torch.mean(y * torch.log(probs))


class ClassifierHandler:

    @staticmethod
    def run_validation(test_dataloader, epoch, music_classifier):
        # validation
        n_classes = len(label2genre)
        val_tq = tqdm(test_dataloader, desc=f'VALIDATION: Epoch {epoch}')
        all_hits = None
        all_genre_hits = [None for _ in range(n_classes)]
        all_losses = []
        for raw_features in val_tq:  # [b, T]
            loss, hits, genre_hits = music_classifier.validation_forward(raw_features)

            all_losses.append(loss)
            if all_hits is None:
                all_hits = hits
            else:
                all_hits = torch.concat((all_hits, hits))

            for c in range(n_classes):
                if all_genre_hits[c] is None:
                    all_genre_hits[c] = genre_hits[c]
                else:
                    all_genre_hits[c] = torch.concat((all_genre_hits[c], genre_hits[c]))

        print(f'\nloss: {np.mean(all_losses)}  acc: {all_hits.float().mean()}')

        for c in range(n_classes):
            print(f'genre: {c}  accuracy: {all_genre_hits[c].float().mean()}')

    @staticmethod
    def train_new_model(training_parameters: TrainingParameters) -> MusicClassifier:
        """
        This function should create a new 'MusicClassifier' object and train it from scratch.
        You could program your training loop / training manager as you see fit.
        """
        opt_params = OptimizationParameters()
        train_dataset = Mp3Dataset("jsons/train.json")
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=training_parameters.batch_size,
            shuffle=True)

        test_dataset = Mp3Dataset("jsons/test.json")
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=training_parameters.batch_size,
            shuffle=False)

        music_classifier = MusicClassifier(opt_params)

        music_classifier.estimate_feats_distribution(train_dataloader)
        for epoch in range(training_parameters.num_epochs):

            tq = tqdm(train_dataloader, desc=f'Epoch {epoch}')
            for raw_features in tq: # [b, T]
                loss, accuracy = music_classifier.train_step(raw_features)

                tq.set_postfix(loss=f"{loss:.4f}", accuracy=f"{accuracy:.4f}",
                               W_norm=f"{music_classifier.W.norm():.4f}",
                               b_norm=f"{music_classifier.b.norm():.4f}",
                               grad_norm=f"{music_classifier.grad.norm():.4f}")

            ClassifierHandler.run_validation(test_dataloader, epoch, music_classifier)

        music_classifier.save("model")

    @staticmethod
    def get_pretrained_model() -> MusicClassifier:
        """
        This function should construct a 'MusicClassifier' object, load it's trained weights / 
        hyperparameters and return the loaded model
        """
        music_classifier = MusicClassifier()
        music_classifier.load("model")
        return music_classifier

    