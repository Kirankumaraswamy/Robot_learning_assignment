from __future__ import print_function

import sys

import torch

import utils

sys.path.append("../")

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.utils import shuffle, resample

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.
    X_train = rgb2gray(X_train)/255.0
    X_valid = rgb2gray(X_valid)/255.0

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    #Append the history into the input array
    if history_length >= 1:
        X_train = np.append([X_train[0] for i in range(history_length)], X_train, 0)
        X_valid = np.append([X_valid[0] for i in range(history_length)], X_valid, 0)

    X_train = np.array([X_train[i: i + history_length + 1] for i in range(len(X_train) - history_length)])
    X_valid = np.array([X_valid[i: i + history_length + 1] for i in range(len(X_valid) - history_length)])

    y_train = np.array([action_to_id(action) for action in y_train])
    y_valid = np.array([action_to_id(action) for action in y_valid])

    print("Upsampling training data")
    X_train_0 = X_train[y_train == 0]
    X_train_1 = X_train[y_train == 1]
    X_train_2 = X_train[y_train == 2]
    X_train_3 = X_train[y_train == 3]
    X_train_4 = X_train[y_train == 4]

    X_train = np.concatenate([
        X_train_0,
        resample(X_train_1, replace=True, n_samples=len(X_train_1) * 2),
        resample(X_train_2, replace=True, n_samples=len(X_train_2) * 3),
        resample(X_train_3, replace=True, n_samples=len(X_train_3) * 2),
        resample(X_train_4, replace=True, n_samples=len(X_train_4) * 6)
    ])

    y_train = np.concatenate([
        np.zeros((len(X_train_0))),
        np.ones((len(X_train_1) * 2)),
        np.ones(len(X_train_2) * 3) * 2,
        np.ones(len(X_train_3) * 2) * 3,
        np.ones(len(X_train_4) * 6) * 4
    ])

    print("Shuffle data")
    X_train, y_train = shuffle(X_train, y_train)

    return X_train, y_train, X_valid, y_valid

def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, history_length=0, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  

    print("... train model")

    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    agent = BCAgent(input_shape = X_train.shape, history_length=history_length)
    
    tensorboard_eval = Evaluation(tensorboard_dir, "bc_history5", ["training_accuracy", "validation_accuracy"])

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser

    n_train_batches = int(X_train.shape[0] / batch_size)
    n_valid_batches = int(X_valid.shape[0] / batch_size)
    epochs =75
    # training loop
    for i in range(epochs):
        train_loss = 0
        training_acc = []
        training_loss = []
        validation_acc = []
        validation_loss = []

        for j in range(n_train_batches):
            if j*batch_size+batch_size < len(X_train):
                X_batch = X_train[j*batch_size: j*batch_size+batch_size]
                y_batch = y_train[j*batch_size: j*batch_size+batch_size]

            else:
                X_batch = X_train[j * batch_size:]
                y_batch = y_train[j * batch_size:]

            train_loss, training_prediction = agent.update(X_batch, y_batch, type="train")
            training_prediction = torch.max(training_prediction.data, 1)[1].cpu().numpy()
            training_acc.append(f1_score(training_prediction, y_batch, average="weighted"))
            training_loss.append(train_loss.item())

        for j in range(n_valid_batches):
            if j*batch_size+batch_size < len(X_valid):
                X_valid_batch = X_valid[j * batch_size: j * batch_size + batch_size]
                y_valid_batch = y_valid[j * batch_size: j * batch_size + batch_size]

            else:
                X_valid_batch = X_valid[j * batch_size:]
                y_valid_batch = y_valid[j * batch_size:]

            valid_loss, valid_prediction = agent.update(X_valid_batch, y_valid_batch, type="validation")
            valid_prediction = torch.max(valid_prediction.data, 1)[1].cpu().numpy()
            validation_acc.append(f1_score(valid_prediction, y_valid_batch, average="weighted"))
            validation_loss.append(valid_loss.item())


        episode_training_acc = np.mean(np.array(training_acc))
        episode_training_loss = np.mean(np.array(training_loss))
        episode_validation_acc = np.mean(np.array(validation_acc))
        episode_validation_loss = np.mean(np.array(validation_loss))
        if i % 5 ==0:
            tensorboard_eval.write_episode_data(i+1, {"training_accuracy": episode_training_acc, "validation_accuracy": episode_validation_acc})
        print(f"Epoch: {i+1}\tTrain Loss: {episode_training_loss:.3f}\tTrain f1_score:{episode_training_acc:.3f}\tValidation f1_score:{episode_validation_acc:.3f}")
      
    # TODO: save your agent
    save_model_str = os.path.join(model_dir, "bc_agent_history5.pt")
    model_dir = agent.save(save_model_str)
    print("Model saved in file: %s" % save_model_str)


if __name__ == "__main__":

    history_length = 5
    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=history_length)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=1000, batch_size=64, lr=0.01, history_length=history_length)
 
