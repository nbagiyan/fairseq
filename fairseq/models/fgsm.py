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
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

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

        loss = self.loss_fn(scores, y_var.unsqueeze(1))

        loss.backward()

        grad_sign = X_var.grad.data.cpu().sign().numpy()

        if adversarial_target is None:

            X += self.epsilon * grad_sign

        else:

            X -= self.epsilon * grad_sign

        return X
