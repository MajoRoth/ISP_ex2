import torch

from genre_classifier import *
import torchaudio

if __name__ == "__main__":

    handler = ClassifierHandler()

    # check that training is working
    TRAIN = True
    if TRAIN:
        training_params = TrainingParameters(batch_size=32, num_epochs=20)
        try:
            handler.train_new_model(training_params)
            print("Train dummy test passed")
        except Exception as e:
            print(f"Train dummy test failed, exception:\n{e}")

    # check that model object is obtained
    try:
        music_classifier = handler.get_pretrained_model()
        print("Get pretrained object dummy test passed")
    except Exception as e:
        print(f"Get pretrained object dummy test failed, exception:\n{e}")

    # check that classification works
    try:
        y1, sr1 = torchaudio.load('parsed_data/heavy-rock/test/4.mp3')
        y2, sr2 = torchaudio.load('parsed_data/classical/test/7.mp3')
        y = torch.stack((y1, y2), dim=0)
        preds = music_classifier.classify(y)
        print("Classify test passed")
    except Exception as e:
        print(f"Classify test failed, exception:\n{e}")



    