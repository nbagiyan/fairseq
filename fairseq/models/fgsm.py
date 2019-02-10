import torch.nn as nn
import torch
import numpy as np


class FGSMAttack(object):
    def __init__(self, model, epsilon):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, adversarial_target=None):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        if adversarial_target is None:
            y_var = torch.LongTensor(y)
        else:
            y_var = torch.LongTensor(adversarial_target)

        X = np.copy(X_nat)

        X_var = torch.tensor(X, requires_grad=True)

        scores = self.model(X_var)

        loss = self.loss_fn(scores, y_var)

        loss.backward()

        grad_sign = X_var.grad.data.cpu().sign().numpy()

        if adversarial_target is None:

            X += self.epsilon * grad_sign

        else:

            X -= self.epsilon * grad_sign

        return X


class Classificator(nn.Module):

    def __init__(self, hidden_dim):
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, input):
        return self.block(input)
