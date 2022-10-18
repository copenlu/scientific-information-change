from transformers import Trainer
import torch


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        weight = kwargs.pop('weight')
        super().__init__(**kwargs)
        self.weight = weight.cuda()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss