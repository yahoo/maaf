# Copyright 2022 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

import torch
import torchvision
from torch.nn import functional as tfunc


# Loading weights Auxiliary functions
def remove_prefix(prefix, a_string):
    l_prefix = len(prefix)
    if a_string[:l_prefix] == prefix:
        final_string = a_string[l_prefix:]
    else:
        final_string = a_string
    return(final_string)


def load_pretrained_weights(model, weights_path, freeze, prefix_to_remove):
    model_dict = model.state_dict()
    saved_state_dict = torch.load(weights_path)['state_dict']
    print("Loading image model weights from: %s" % weights_path)
    # 1. filter out unnecessary keys
    pretrained_dict = {remove_prefix(prefix_to_remove, k): v
        for k, v in saved_state_dict.items() if remove_prefix(prefix_to_remove, k) in model_dict}
    # 2. overwrite entries in the existing state dict
    for tensor_name in pretrained_dict.keys():
        print('Loading %s' % tensor_name)
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    if freeze:
        for name, param in model.named_parameters():
            if name in pretrained_dict:
                print('freezing parameter:  %s' % name)
                param.requires_grad = False
        model.eval()


def repeating_eye(in_channels, out_channels):
    repetitions = in_channels // out_channels
    eye = torch.eye(out_channels)
    return eye.repeat(1, repetitions)


class ConvProjection(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1,
                 dtype=None, initialization=None, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)

        if initialization == "identity":
            assert kernel_size == 1
            eyes = repeating_eye(in_channels, out_channels)
            self.conv.weight.data = eyes.reshape(out_channels, in_channels, 1, 1)
        elif initialization is not None:
            raise ValueError(f"Unsupported initialization {initialization}")

        if dtype is not None:
            self.conv = self.conv.to(dtype)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = tfunc.relu(x, inplace=True)
        return x.view(x.shape[0], x.shape[1], -1).transpose(-2, -1)


class GlobalAvgPool2d(torch.nn.Module):

    def forward(self, x):
        return tfunc.adaptive_avg_pool2d(x, (1, 1))


class ResNet(torch.nn.Module):
    """
    ResNet, possibly returning intermediate layer outputs, possibly projected
    forward returns a dict
    """

    def __init__(self, architecture="resnet50", out_features=["fc"],
                 out_channels=None, pretrained=True, dict_out=False):
        """
        Args:
        architecture (str): resnet50 or resnet18
        out_features (list[str | int]): layers whose outputs to return,
            from among "stem", 1, 2, 3, 4, "fc"
        out_channels (None | int): if None, outputs returned directly from
            resnet. If an int, learnable convs project outputs to out_channels
        pretrained (bool): if True, load torchvision's ImageNet-trained weights
            Note this requires internet or precaching these weights
        """
        super().__init__()
        if architecture == 'resnet50':
            top_channel = 2048
            print("Using ResNet50")
            self.model = torchvision.models.resnet50(pretrained=pretrained)
        elif architecture == 'resnet18':
            top_channel = 512
            print("Using ResNet18")
            self.model = torchvision.models.resnet18(pretrained=pretrained)
        else:
            raise ValueError("Invalid image_model_arch {}".format(
                             architecture))

        self.out_features = out_features

        # drop any layers that aren't being used
        labels_to_inds = {"stem": 0, 1: 1, 2: 2, 3: 3, 4: 4, "fc": 5}
        # inds_to_labels = {val: key for key, val in labels_to_inds.items()}
        inds = [labels_to_inds[label] for label in self.out_features]
        last_ind = max(inds)
        self.layers = []
        for ii in range(1, 5):
            if ii > last_ind:
                delattr(self.model, f"layer{ii}")
            else:
                self.layers.append((ii, getattr(self.model, f"layer{ii}")))

        if out_channels is not None:
            self.projections = torch.nn.ModuleDict()
            for ii, layer in self.layers:
                if ii in self.out_features:
                    in_channel = top_channel // (2**(4 - ii))
                    self.projections[str(ii)] = \
                        ConvProjection(in_channel, out_channels, kernel_size=1)

            if "fc" in self.out_features:
                self.model.avgpool = GlobalAvgPool2d()
                self.avgpool = self.model.avgpool

                self.model.fc = torch.nn.Sequential(
                    torch.nn.Linear(self.model.fc.weight.shape[1],
                                    out_channels))
                if len(self.out_features) == 1:
                    self.projections = None
        else:
            self.projections = None

        if "fc" in out_features:
            self.fc = self.model.fc
        else:
            del self.model.fc

    def pretrained_parameters(self):
        if self.projections is not None:
            scratch = set([param for param in self.projections.parameters()])
            all_param = set([param for param in self.parameters()])
            return all_param.difference(scratch)
        else:
            return self.parameters()

    def forward(self, imgs):
        """Returns: {layer_name: output}"""
        out = {}
        xx = imgs

        xx = self.model.conv1(xx)
        xx = self.model.bn1(xx)
        xx = self.model.relu(xx)
        xx = self.model.maxpool(xx)
        if "stem" in self.out_features:
            out["stem"] = xx

        for ind, layer in self.layers:
            xx = layer(xx)
            if ind in self.out_features:
                out[ind] = xx

        if "fc" in self.out_features:
            xx = self.avgpool(xx)
            out["fc"] = self.fc(xx.view(xx.size(0), -1))

        if self.projections is not None:
            proj = [self.projections[str(ii)](out[ii])
                    for ii, layer in self.layers if ii in out]
            if "fc" in out:
                proj += [out["fc"].unsqueeze(1)]
            out["projections"] = torch.cat(proj, dim=1)

        return out

    def get_num_tokens(self):
        num = 0
        if 2 in self.out_features:
            num += 28**2
        if 3 in self.out_features:
            num += 14**2
        if 4 in self.out_features:
            num += 7**2
        if "fc" in self.out_features:
            num += 1
        return num

    def resolutionwise_pool(self, xx):
        """Pool over space at each resolution, then average results."""
        resolutions = []
        start = 0
        if 2 in self.out_features:
            x2 = xx[:, :28**2]
            resolutions.append(x2)
            start = 28**2
        if 3 in self.out_features:
            x3 = xx[:, start:start+14**2]
            resolutions.append(x3)
            start += 14**2
        if 4 in self.out_features:
            x4 = xx[:, start:start+7**2]
            resolutions.append(x4)
            start += 7**2
        if "fc" in self.out_features:
            xfc = xx[:, start:]
            resolutions.append(xfc)

        resmeans = []
        for res in resolutions:
            resmeans.append(torch.mean(res, 1))

        return torch.mean(torch.stack(resmeans), 0)


def build_image_model(cfg):
    architecture = cfg.MODEL.IMAGE_MODEL.ARCHITECTURE
    if architecture is None:
        return None

    out_features = cfg.MODEL.IMAGE_MODEL.OUTPUTS
    # if cfg.MODEL.COMPOSITION in MAAF_ALIASES:
    #     out_channels = cfg.MODEL.EMBED_DIM
    # else:
    #     out_channels = None
    out_channels = cfg.MODEL.EMBED_DIM
    pretrained = cfg.MODEL.IMAGE_MODEL.PRETRAINED and \
        cfg.MODEL.IMAGE_MODEL.WEIGHTS is None and \
        cfg.MODEL.WEIGHTS is None
    img_model = ResNet(architecture, out_features, out_channels=out_channels,
                 pretrained=pretrained)

    if cfg.MODEL.IMAGE_MODEL.WEIGHTS is not None:
        # saved_state_dict = torch.load(opt.image_model_path)['state_dict']
        # self.model.load_state_dict(saved_state_dict)
        load_pretrained_weights(
            model=img_model, weights_path=cfg.MODEL.IMAGE_MODEL.WEIGHTS,
            freeze=cfg.MODEL.IMAGE_MODEL.FREEZE_WEIGHTS,
            prefix_to_remove='img_model.')

    if cfg.MODEL.IMAGE_MODEL.FREEZE_WEIGHTS:
        print("Freezing Image model weights")
        for param in img_model.parameters():
            param.requires_grad = False

    return img_model
