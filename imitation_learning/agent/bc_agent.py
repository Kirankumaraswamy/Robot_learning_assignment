import torch
from agent.networks import CNN
import torch.nn as nn
from sklearn.metrics import f1_score

class BCAgent:
    
    def __init__(self, input_shape, lr=1e-3, num_classes=5, history_length=0):
        # TODO: Define network, loss function, optimizer
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = CNN(input_shape, num_classes, history_length).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)


    def update(self, X_batch, y_batch, type="train"):
        if type == "train":
            self.net.train()
            X_batch = torch.Tensor(X_batch).to(self.device)
            y_batch = torch.LongTensor(y_batch).to(self.device)

            predict = self.predict(X_batch)
            loss = self.criterion(predict, y_batch).to(self.device)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            self.net.eval()
            with torch.no_grad():
                X_batch = torch.Tensor(X_batch).to(self.device)
                y_batch = torch.LongTensor(y_batch).to(self.device)

                predict = self.predict(X_batch)
                loss = self.criterion(predict, y_batch).to(self.device)

        return loss, predict

    def predict(self, X):
        # TODO: forward pass
        outputs = self.net(X)
        return outputs

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))
