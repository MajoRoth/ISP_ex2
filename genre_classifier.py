from abc import abstractmethod

import numpy as np
import torch
from enum import Enum
import typing as tp
from dataclasses import dataclass
# from dataset import Mp3Dataset
from tqdm import tqdm


from torch.utils.data import Dataset
import json
import librosa

from features import *


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
        y, sr = librosa.load(self.files[index]['path'])
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
    num_epochs: int = 100
    train_json_path: str = "jsons/train.json" # you should use this file path to load your train data
    test_json_path: str = "jsons/test.json" # you should use this file path to load your test data
    # other training hyper parameters


@dataclass
class OptimizationParameters:
    """
    This dataclass defines optimization related hyper-parameters to be passed to the model.
    feel free to add/change it as you see fit.
    """
    learning_rate: float = 0.005


class MusicClassifier:
    """
    You should Implement your classifier object here
    """

    def __init__(self, opt_params: OptimizationParameters = None, encoding_dim=5461, **kwargs):
        """
        This defines the classifier object.
        - You should define your weights and biases as class components here.
        - You could use kwargs (dictionary) for any other variables you wish to pass in here.
        - You should use `opt_params` for your optimization and you are welcome to experiment
        """
        self.opt_params = opt_params
        self.encoding_dim = encoding_dim  # input size
        self.num_classes = len(label2genre)  # output size
        INIT_STD = 0.05
        self.W = torch.randn((self.encoding_dim, self.num_classes)) * INIT_STD
        self.b = torch.zeros((1, self.num_classes))

        self.loss = list()
        self.data_sr = 22050

    def exctract_feats(self, wavs: torch.Tensor):
        """
        this function extract features from a given audio.
        we will not be observing this method.

        Return: [batch, encoding_dim]
        """
        RMS_EMPIRICAL_MEAN = 0.1
        RMS_EMPIRICAL_STD = 0.07
        ZCR_EMPIRICAL_MEAN = 0.1
        ZCR_EMPIRICAL_STD = 0.1
        MFCC_EMPIRICAL_MEAN = -2.0
        MFCC_EMPIRICAL_STD = 60.0

        zcr = (zero_crossing_rate(wavs) - ZCR_EMPIRICAL_MEAN) / ZCR_EMPIRICAL_STD
        rms_per_window = (extract_rms(wavs) - RMS_EMPIRICAL_MEAN) / RMS_EMPIRICAL_STD
        mfccs = (extract_mfccs(wavs, self.data_sr) - MFCC_EMPIRICAL_MEAN) / MFCC_EMPIRICAL_STD
        feats = torch.concat((zcr.unsqueeze(1), rms_per_window, mfccs), dim=-1)
        return feats


    def forward(self, feats: torch.Tensor) -> tp.Any:
        """
        this function performs a forward pass throuh the model, outputting scores for every class.
        feats: batch of extracted faetures
        """
        # feats [b, d]
        # W [d, 3]
        Z = torch.matmul(feats, self.W) + self.b
        return torch.softmax(Z, dim=-1) #MusicClassifier.softmax(Z)


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
        y_labeled = torch.nn.functional.one_hot(labels)
        error = y_labeled - output_scores
        batch = feats.shape[0]
        feats = torch.concat((feats, torch.ones((batch, 1))), dim=-1) # adding ones for bias term
        self.grad = (1 / float(batch)) * -torch.matmul(error.T, feats)
        self.W -= self.opt_params.learning_rate * self.grad[:, :-1].T
        self.b -= self.opt_params.learning_rate * self.grad[:, -1].unsqueeze(0)

    # def validation_step(self, raw_features):
    #     features = self.exctract_feats(raw_features['waveform'])
    #     y_preds = self.forward(features)

    def train_step(self, raw_features):
        features = self.exctract_feats(raw_features['waveform'])
        y_preds = self.forward(features)

        labels_onehot = torch.nn.functional.one_hot(raw_features['label'], num_classes=len(label2genre))

        loss = -torch.log((y_preds * labels_onehot).sum(dim=-1)).mean()
        accuracy = (torch.argmax(y_preds, dim=-1) == raw_features['label']).float().mean()

        self.backward(features, output_scores=y_preds, labels=raw_features['label'])

        return loss, accuracy

    def validation_forward(self, raw_features):
        features = self.exctract_feats(raw_features['waveform'])
        y_preds = self.forward(features)

        n_classes = len(label2genre)
        labels_onehot = torch.nn.functional.one_hot(raw_features['label'], num_classes=n_classes)

        loss = -torch.log((y_preds * labels_onehot).sum(dim=-1)).mean()
        hits = torch.argmax(y_preds, dim=-1) == raw_features['label']

        genre_hits = [None for _ in range(n_classes)]
        for c in range(n_classes):
            genre_hits[c] = hits[raw_features['label'] == c]



        # accuracy per class
        # TODO
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
        m = {'W': self.W, 'b': self.b, 'opt': OptimizationParameters}
        torch.save(m, f"model_files/{name}.pt")

    def load(self, name: str):
        loaded = torch.load(f"model_files/{name}.pt")
        self.W = loaded["W"]
        self.b = loaded["b"]
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

        print(f'loss: {np.mean(all_losses)}  acc: {all_hits.float().mean()}')

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

    