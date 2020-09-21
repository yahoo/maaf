# Copyright 2020 Verizon Media, Licensed under the terms of the Apache License, Version 2.0.
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
    print("Loading image model weights from: %s"% weights_path)
    # 1. filter out unnecessary keys
    pretrained_dict = {remove_prefix(prefix_to_remove, k): v
        for k, v in saved_state_dict.items() if remove_prefix(prefix_to_remove, k) in model_dict}
    # 2. overwrite entries in the existing state dict
    for tensor_name in pretrained_dict.keys():
        print ('Loading %s'%tensor_name)
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    if freeze:
        for name,param in model.named_parameters():
            if name in pretrained_dict:
               print('freezing parameter:  %s'%name)
               param.requires_grad = False
        model.eval()


class BasicConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                    bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return tfunc.relu(x, inplace=True)


class GlobalAvgPool2d(torch.nn.Module):

    def forward(self, x):
        return tfunc.adaptive_avg_pool2d(x, (1, 1))


class ImageModelBase(torch.nn.Module):

    def __init__(self, model, opt):
        super(ImageModelBase, self).__init__()
        self.opt = opt
        self.model = model
        if opt.image_model_path is not None:
            # saved_state_dict = torch.load(opt.image_model_path)['state_dict']
            # self.model.load_state_dict(saved_state_dict)
            load_pretrained_weights(
                model=model, weights_path=opt.image_model_path,
                freeze=opt.freeze_img_model, prefix_to_remove='img_model.')

        if opt.freeze_img_model:
            print("Freezing Image model weights")
            for param in self.model.parameters():
                param.requires_grad = False

        # replaces existing attributes
        self.model.avgpool = GlobalAvgPool2d()
        self.avgpool = self.model.avgpool

        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(self.base_network_embedding_size,
                            opt.embed_dim))
        self.fc = self.model.fc

    def forward(self, xx):
        return self.model(xx)


class ResNetBase(ImageModelBase):

    def __init__(self, opt):
        if opt.image_model_arch == 'resnet50':
            print("Using ResNet50")
            model = torchvision.models.resnet50(pretrained=not opt.not_pretrained)
            self.base_network_embedding_size = 2048
        elif opt.image_model_arch == 'resnet18':
            print("Using ResNet18")
            model = torchvision.models.resnet18(pretrained=not opt.not_pretrained)
            self.base_network_embedding_size = 512
        else:
            raise ValueError("Invalid image_model_arch {}".format(
                             opt.image_model_arch))
        super(ResNetBase, self).__init__(model, opt)


class ResNetSpatial(ResNetBase):
    """
    Base class for ResNet with outputs projected from
    layers with spatial dimensions.
    """

    def __init__(self, opt):
        super(ResNetSpatial, self).__init__(opt)
        if opt.image_model_arch == "resnet18":
            l2_emb_size = 128
            l3_emb_size = 256
            l4_emb_size = 512
        elif opt.image_model_arch == "resnet50":
            l2_emb_size = 512
            l3_emb_size = 1024
            l4_emb_size = 2048
        else:
            raise ValueError("Invalid image_model_arch {}".format(
                             opt.image_model_arch))
        print("spatial")

        embed_dim = self.opt.embed_dim
        if "2" in self.opt.att_layer_spec:
            self.l2_conv_proj = BasicConv2d(l2_emb_size, embed_dim,
                                            kernel_size=1)
        if "3" in self.opt.att_layer_spec:
            self.l3_conv_proj = BasicConv2d(l3_emb_size, embed_dim,
                                            kernel_size=1)
        if "4" in self.opt.att_layer_spec:
            self.l4_conv_proj = BasicConv2d(l4_emb_size, embed_dim,
                                            kernel_size=1)

    def forward(self, imgs):
        xx = imgs

        xx = self.model.conv1(xx)
        xx = self.model.bn1(xx)
        xx = self.model.relu(xx)
        xx = self.model.maxpool(xx)

        xx = self.model.layer1(xx)
        xx = self.model.layer2(xx)
        xx2 = xx
        xx = self.model.layer3(xx)
        xx3 = xx
        xx = self.model.layer4(xx)
        xx4 = xx

        embed_dim = self.opt.embed_dim

        if "2" in self.opt.att_layer_spec:
            x2_proj = self.l2_conv_proj(xx2).view(-1, embed_dim, 28**2)
            x2_proj = x2_proj.transpose(-2, -1)
        else:
            x2_proj = None
        if "3" in self.opt.att_layer_spec:
            x3_proj = self.l3_conv_proj(xx3).view(-1, embed_dim, 14**2)
            x3_proj = x3_proj.transpose(-2, -1)
        else:
            x3_proj = None
        if "4" in self.opt.att_layer_spec:
            x4_proj = self.l4_conv_proj(xx4).view(-1, embed_dim, 7**2)
            x4_proj = x4_proj.transpose(-2, -1)
        else:
            x4_proj = None

        nums = [2, 3, 4]
        all_proj = [x2_proj, x3_proj, x4_proj]
        concat = [prj for ii, prj in enumerate(all_proj)
                  if str(nums[ii]) in self.opt.att_layer_spec]

        if len(concat) == 0:
            return xx4
        elif len(concat) == 1:
            return concat[0]
        else:
            return torch.cat(concat, dim=1)

    def final_layers(self, xx):
        xx = self.avgpool(xx)
        xx = xx.view(xx.size(0), -1)
        return self.fc(xx)

    def get_num_tokens(self):
        num = 0
        if "2" in self.opt.att_layer_spec:
            num += 28**2
        if "3" in self.opt.att_layer_spec:
            num += 14**2
        if "4" in self.opt.att_layer_spec:
            num += 7**2
        return num

    def resolutionwise_pool(self, xx):
        """Pool over space at each resolution, then average results."""
        resolutions = []
        if "2" in self.opt.att_layer_spec:
            x2 = xx[:, :28**2]
            resolutions.append(x2)
        if "3" in self.opt.att_layer_spec:
            start = 28**2 if "2" in self.opt.att_layer_spec else 0
            x3 = xx[:, start:start+14**2]
            resolutions.append(x3)
        if "4" in self.opt.att_layer_spec:
            x4 = xx[:, -7**2:]
            resolutions.append(x4)

        resmeans = []
        for res in resolutions:
            resmeans.append(torch.mean(res, 1))

        return torch.mean(torch.stack(resmeans), 0)
