import torch.nn as nn
import torch
import numpy as np


class FGSMAttack(object):

    def __init__(self, model, epsilon, num_iter=5, momentum=0.1):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.num_iter = num_iter
        self.momentum = momentum

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

        alpha = self.epsilon / self.num_iter

        momentum = 0

        for _ in range(self.num_iter):

            X_var = torch.tensor(X, requires_grad=True)

            scores = self.model(X_var)

            loss = self.loss_fn(scores, y_var.unsqueeze(1).float())

            loss.backward()

            grad = X_var.grad.data.cpu().numpy()

            grad_norm = np.linalg.norm(grad, ord=1)

            momentum = self.momentum * momentum + (1 - self.momentum) * grad/grad_norm

            if adversarial_target is None:

                X += alpha * np.sign(momentum)

            else:

                X -= alpha * np.sign(momentum)

        return X
