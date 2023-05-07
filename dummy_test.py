from genre_classifier import *

if __name__ == "__main__":

    handler = ClassifierHandler()

    # check that training is working
    training_params = TrainingParameters(batch_size=1, num_epochs=1)
    try:
        handler.train_new_model(training_params)
        print("Train dummy test passed")
    except Exception as e:
        print(f"Train dummy test failed, exception:\n{e}")

    # check that model object is obtained
    try:
        handler.get_pretrained_model()
        print("Get pretrained object dummy test passed")
    except Exception as e:
        print(f"Get pretrained object dummy test failed, exception:\n{e}")

    # feel free to add tests here. 
    # We will not be giving score to submitted tests.
    # You may (and recommended to) share tests with one another.

    