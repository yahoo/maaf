from abc import ABC, abstractmethod
import torch
from .loss import build_loss


class TaskHead(torch.nn.Module, ABC):

    def __init__(self):
        torch.nn.Module.__init__(self)

    @abstractmethod
    def forward(self, images, texts):
        pass

    @abstractmethod
    def compute_loss(self, source, target=None, labels=None):
        pass


class Classification(TaskHead):

    def __init__(self, loss, embed_dim=512, num_classes=3):
        super().__init__()
        self.classification_head = torch.nn.Linear(embed_dim, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss = loss

    def forward(self, composed):
        return self.classification_head(composed)

    def probabilities(self, composed):
        logits = self(composed)
        return self.softmax(logits)

    def compute_loss(self, source, target=None, labels=None):
        logits = self(source)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).sum().item() / len(labels)
        loss_value = self.loss(logits, labels)
        metrics = {"loss": loss_value.item(),
                   "accuracy": accuracy}
        return loss_value, metrics


class Regression(TaskHead):

    def __init__(self, loss, embed_dim=512):
        super().__init__()
        self.regression_head = torch.nn.Linear(embed_dim, 1)
        self.loss = loss

    def forward(self, composed):
        return torch.sigmoid(self.regression_head(composed))

    def compute_loss(self, source, target=None, labels=None):
        output = self(source)
        loss_value = self.loss(output, labels.float())
        metrics = {"loss": loss_value.item()}
        return loss_value, metrics


class NormalizationLayer(torch.nn.Module):
    """Class for normalization layer."""

    def __init__(self, normalize_scale=1.0, learn_scale=True):
        super().__init__()
        self.norm_s = torch.log(torch.FloatTensor([normalize_scale]))
        if learn_scale:
            self.norm_s = torch.nn.Parameter(self.norm_s)
        self.epsilon = 1e-9

    def forward(self, x):
        norm = torch.norm(x, dim=1, keepdim=True).expand_as(x)
        factor = torch.exp(self.norm_s)
        features = factor * x / (norm + self.epsilon)
        return features


class Metric(TaskHead):

    def __init__(self, loss, initial_normalization_factor=4.0):
        super().__init__()
        self.loss = loss
        self.normalization_layer = NormalizationLayer(
            normalize_scale=initial_normalization_factor, learn_scale=True)

    def forward(self, composed):
        return self.normalization_layer(composed)

    def compute_loss(self, source, target, labels=None):
        source_emb = self.forward(source)
        target_emb = self.forward(target)

        assert source_emb.shape[1] == target_emb.shape[1]
        loss_value = self.loss(source_emb, target_emb, labels=labels)
        metrics = {"loss": loss_value.item()}
        if torch.isnan(loss_value):
            import IPython; IPython.embed()
        return loss_value, metrics


def get_task_head(cfg):
    loss_obj, task = build_loss(cfg)

    if task == "metric":
        head = Metric(loss_obj, cfg.MODEL.INITIAL_NORMALIZATION_FACTOR)
    elif task == "regression":
        head = Regression(loss_obj, cfg.MODEL.EMBED_DIM)
    else:
        head = Classification(loss_obj, embed_dim=cfg.MODEL.EMBED_DIM,
                              num_classes=cfg.DATASET.NUM_CLASSES)

    return head, task
