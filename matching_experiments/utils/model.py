import torch
from torch import nn
from torch.optim import SGD
from transformers import AutoModel
from tqdm import tqdm
import torch.nn.functional as F
import ipdb


class GradientReversal(torch.autograd.Function):
    """
    Basic layer for doing gradient reversal
    """
    lambd = 1.0
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReversal.lambd * grad_output.neg()



class DomainAdversarialModel(nn.Module):
    """
    A really basic wrapper around BERT
    """
    def __init__(self, model: AutoModel, n_classes: int = 2, **kwargs):
        super(DomainAdversarialModel, self).__init__()

        self.model = AutoModel.from_pretrained(model)
        self.domain_classifier = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            labels: torch.LongTensor = None,
            **kwargs
    ):

        # 1) Get the CLS representation from BERT
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask
        )
        # (b x n_classes)
        cls_hidden_state = outputs.pooler_output

        adv_input = GradientReversal.apply(cls_hidden_state)

        adv_logits = self.domain_classifier(adv_input)

        outputs['logits'] = adv_logits

        loss_fn = nn.CrossEntropyLoss()
        if labels is not None:
            loss = loss_fn(adv_logits, labels)
            outputs['loss'] = loss

        return outputs

    def save_pretrained(self, output_dir: str):
        self.model.save_pretrained(output_dir)

# Optimize the softmax temperature to minimize the negative log likelihood
class Temp(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))
    def forward(self, logits):
        return logits / self.T


def calculate_log_likelihood(model, loader, T, device):
    # loader = torch.utils.data.DataLoader(dset, batch_size=32,
    #                                           num_workers=2)
    with torch.no_grad():
        labels_all = []
        preds_all = []
        for i, batch in enumerate(tqdm(loader), 0):
            # get the inputs; data is a list of [inputs, labels]
            for b in batch:
                batch[b] = batch[b].to(device)
            labels = batch.pop('labels')
            # forward + backward + optimize
            outputs = model(**batch)
            logits = outputs['logits']
            logits /= T
            preds = F.log_softmax(logits, dim=-1)
            labels_all.append(labels.detach())
            preds_all.append(preds.detach())
        nll = F.nll_loss(torch.concat(preds_all), torch.concat(labels_all), reduction='mean')

    return nll.item()

def calibrate_temperature(model, loader, device):
    # loader = torch.utils.data.DataLoader(dset, batch_size=32,
    #                                           num_workers=2)

    T = Temp().to(device)
    optim = SGD(T.parameters(), lr=1e-3)
    patience = 10
    c = 0
    eps = 1e-5
    t_curr = 1.0
    done = False
    print(f"NLL before calibration: {calculate_log_likelihood(model, loader, t_curr, device)}")
    for epoch in range(3):  # loop over the dataset multiple times
        for i, batch in enumerate(tqdm(loader), 0):
            # get the inputs; data is a list of [inputs, labels]
            for b in batch:
                batch[b] = batch[b].to(device)
            labels = batch.pop('labels')
            # zero the parameter gradients
            optim.zero_grad()

            # forward + backward + optimize
            outputs = model(**batch)
            logits = outputs['logits']
            logits = T(logits)
            preds = F.log_softmax(logits, dim=-1)
            nll = F.nll_loss(preds, labels, reduction='mean')
            nll.backward()
            optim.step()
            if abs(t_curr - T.T.item()) > eps:
                c = 0
            else:
                c += 1
                if c == patience:
                    done = True
                    break
            t_curr = T.T.item()
        if done:
            break
    print(f"NLL after calibration: {calculate_log_likelihood(model, loader, t_curr, device)}")
    return t_curr