# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


import clip
from clip.clip import _tokenizer
import torch
from torchvision import transforms as tvt
from PIL import Image
from .composition_models import Addition, MAAF
from .image_model import ConvProjection


def get_image_transform_for_clip():
    return tvt.Compose([
        tvt.Resize(224, interpolation=Image.BICUBIC),
        tvt.CenterCrop(224),
        tvt.ToTensor(),
        tvt.Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
    ])


def get_augmenting_image_transform_for_clip():
    return tvt.Compose([
        tvt.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.3)),
        tvt.RandomHorizontalFlip(),
        tvt.ToTensor(),
        tvt.Lambda(lambda xx: xx + 0.01 * torch.randn(xx.shape, device="cpu")),
        tvt.Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
    ])


def tokenize(texts, context_length=77):
    """
    Adapted from CLIP code to truncate long sequences rather than raise an exception.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class ClipModel(Addition):

    def __init__(self, loss, image_model='RN50', text_model=None, prompt="",
                 pretrain=True, load_image_transform=True):
        super().__init__(loss)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip, image_transform = clip.load(image_model, device, jit=False)
        self.prompt = prompt
        self.image_model = self.clip.visual
        self.text_model = self.clip.transformer
        if load_image_transform:
            self.image_transform = image_transform

        if not pretrain:
            self.clip.initialize_parameters()

    def extract_img_feature(self, images):
        if all([im is None for im in images]):
            return None
        return self.clip.encode_image(images)

    def extract_text_feature(self, texts):
        if all([tt is None for tt in texts]):
            return None
        tokenized = [tokenize(f"{self.prompt}{tx}") for tx in texts]
        text_inputs = torch.cat(tokenized).to(self.device)
        return self.clip.encode_text(text_inputs)


class ScrambledClip(ClipModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # shuffle the text_projection outputs
        # to destroy alignment with image embedding
        num_rows, num_cols = self.clip.text_projection.data.shape
        col_inds = torch.randperm(num_cols)
        self.clip.text_projection.data = \
            self.clip.text_projection.data[:, col_inds]


class MisalignedClip(ClipModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        other_clip, _ = clip.load("RN101", device, jit=False)
        if not kwargs["pretrain"]:
            other_clip.initialize_parameters()
        self.clip.transformer = other_clip.transformer
        self.text_model = self.clip.transformer
        del other_clip.visual


class ClipMAAF(MAAF):

    def __init__(self, loss, image_model='RN50', text_model=None,
                 prompt="", model_dim=512, num_heads=8, ff_width=256,
                 dropout=0.1, num_blocks=1, position_encodings=None,
                 output="rwpool", img_out_features=["fc"],
                 load_image_transform=True, misalignment=None
                 ):

        if position_encodings == "mixed" or image_model != "RN50":
            raise NotImplementedError

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, image_transform = clip.load('RN50', device, jit=False)
        self.prompt = prompt
        image_model = ImageFeatureExtractor(
            clip_model.visual, img_out_features, model_dim).to(device)
        text_model = TextFeatureExtractor(
            clip_model, model_dim, prompt=prompt, dropout=dropout,
            device=device, dtype=clip_model.visual.conv1.weight.dtype,
            misalignment=misalignment
            ).to(device)

        super().__init__(loss, model_dim=model_dim, num_heads=num_heads,
                         ff_width=ff_width, dropout=dropout,
                         num_blocks=num_blocks,
                         position_encodings=position_encodings,
                         output=output, image_model=image_model,
                         text_model=text_model)

        self.clip = clip_model
        if load_image_transform:
            self.image_transform = image_transform


class ClipResMAAF(ClipMAAF):

    def __init__(self, loss, learn_weight=False,
                 maaf_weight=1., **kwargs):
        super().__init__(loss, **kwargs)
        # pre-sigmoid of weights
        dtype = self.clip.visual.conv1.weight.dtype
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.maaf_weight = torch.Tensor([maaf_weight]).to(dtype).to(device)
        self.maaf_weight.requires_grad = False

    def extract_img_feature(self, imgs):
        if all([im is None for im in imgs]):
            return None
        return self.image_model(imgs)

    def compose(self, img_feat, text_emb):
        if img_feat is None:
            img_maaf_inputs = None
        else:
            img_maaf_inputs = img_feat["projections"]

        maaf_feat = super().compose(img_maaf_inputs, text_emb)

        addition_feat = 0
        if img_feat is not None:
            addition_feat = addition_feat + img_feat["attnpool"]
        if text_emb is not None:
            text_emb, text_mask = text_emb
            # get features from eot embedding
            indices = torch.logical_not(text_mask).to(torch.int).argmax(dim=-1) - 1
            text_feat = text_emb[torch.arange(text_emb.shape[0]), indices]
            addition_feat = addition_feat + text_feat

        return addition_feat + self.maaf_weight * maaf_feat


def repeating_eye(in_channels, out_channels):
    repetitions = in_channels // out_channels
    eye = torch.eye(out_channels)
    return eye.repeat(1, repetitions)


class TextFeatureExtractor(torch.nn.Module):

    def __init__(self, clip_model, out_channels, prompt="", dropout=0.1,
                 device="cuda", dtype=torch.float32, misalignment=None):
        super().__init__()
        self.model = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.prompt = prompt
        self.device = device  # TODO: this is static, not so good
        self.dtype = dtype

        # initialize output layer to the first out_channels
        # channels of the CLIP text_projection, which may be the whole thing
        self.out_layer = torch.nn.Linear(self.model.width, out_channels, bias=False)
        self.out_layer.weight.data = clip_model.text_projection[:, :out_channels].T
        self.out_layer = self.out_layer.to(self.dtype)

        if misalignment == "scramble":
            print("Scrambled the text outputs")
            inds = torch.randperm(out_channels)
            self.out_layer.weight.data = \
                self.out_layer.weight.data[inds]
        elif misalignment == "mismatch":
            print("Using a mismatched text encoder")
            other_clip, _ = clip.load("RN101", device, jit=False)
            # if not kwargs["pretrain"]:
            #     other_clip.initialize_parameters()
            clip_model.transformer = other_clip.transformer
            self.model = clip_model.transformer
            del other_clip.visual

    def pretrained_parameters(self):
        scratch = set([param for param in self.out_layer.parameters()])
        all_param = set([param for param in self.parameters()])
        return all_param.difference(scratch)

    def forward(self, texts):
        tokenized = [tokenize(f"{self.prompt}{tx}") for tx in texts]
        text_inputs = torch.cat(tokenized).to(self.device)
        x = self.token_embedding(text_inputs).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]

        out = self.out_layer(x)

        # clip doesn't seem to use masking for empty token slots, but maaf does
        lengths = text_inputs.argmax(1)  # end token is also highest-# token
        input_mask = torch.zeros(out.shape[:2],
                                 dtype=torch.long, device=x.device)
        for ii in range(len(texts)):
            input_mask[ii, :lengths[ii].item() + 1] = 1

        return out, input_mask

class ImageFeatureExtractor(torch.nn.Module):

    def __init__(self, image_model, out_features, out_channels):
        super().__init__()
        self.model = image_model
        self.out_features = out_features

        top_channel = 2048
        print("Using ResNet50")

        # drop any layers that aren't being used
        labels_to_inds = {"stem": 0, 1: 1, 2: 2, 3: 3, 4: 4, "attnpool": 5}
        # inds_to_labels = {val: key for key, val in labels_to_inds.items()}
        inds = [labels_to_inds[label] for label in self.out_features]
        last_ind = max(inds)
        self.layers = []
        for ii in range(1, 5):
            if ii > last_ind:
                delattr(self.model, f"layer{ii}")
            else:
                self.layers.append((ii, getattr(self.model, f"layer{ii}")))

        self.projections = torch.nn.ModuleDict()
        for ii, layer in self.layers:
            if ii in self.out_features:
                in_channel = top_channel // (2**(4 - ii))
                self.projections[str(ii)] = \
                    ConvProjection(in_channel, out_channels, kernel_size=1,
                                   dtype=self.model.conv1.weight.dtype,
                                   initialization="identity")

        if "attnpool" in self.out_features:
            proj = torch.nn.Linear(
                1024, out_channels, bias=False).to(self.model.conv1.weight.dtype)
            proj.weight.data = repeating_eye(1024, out_channels)
            self.projections["attnpool"] = \
                proj.to(dtype=self.model.conv1.weight.dtype)

    def pretrained_parameters(self):
        scratch = set([param for param in self.projections.parameters()])
        all_param = set([param for param in self.parameters()])
        return all_param.difference(scratch)

    def forward(self, imgs):
        out = {}
        xx = imgs

        def stem(x):
            for conv, bn in [(self.model.conv1, self.model.bn1),
                             (self.model.conv2, self.model.bn2),
                             (self.model.conv3, self.model.bn3)]:
                x = self.model.relu(bn(conv(x)))
            x = self.model.avgpool(x)
            return x

        xx = xx.type(self.model.conv1.weight.dtype)
        xx = stem(xx)
        if "stem" in self.out_features:
            out["stem"] = xx

        for ind, layer in self.layers:
            xx = layer(xx)
            if ind in self.out_features:
                out[ind] = xx

        if "attnpool" in self.out_features:
            xx = self.model.attnpool(xx)
            out["attnpool"] = self.projections["attnpool"](
                xx.view(xx.size(0), -1))

        if self.projections is not None:
            proj = [self.projections[str(ii)](out[ii])
                    for ii, layer in self.layers if ii in out]
            if "attnpool" in out:
                proj += [out["attnpool"].unsqueeze(1)]
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
        if "attnpool" in self.out_features:
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
        if "attnpool" in self.out_features:
            xfc = xx[:, start:]
            resolutions.append(xfc)

        resmeans = []
        for res in resolutions:
            resmeans.append(torch.mean(res, 1))

        return torch.mean(torch.stack(resmeans), 0)


def get_clip_class(cfg):
    kwargs = {"image_model": "RN50", "prompt": cfg.MODEL.CLIP.PROMPT,
              "load_image_transform": cfg.MODEL.INCLUDES_IMAGE_TRANSFORM}

    if cfg.MODEL.COMPOSITION == "clip":
        if cfg.MODEL.CLIP.MISALIGNMENT == "scramble":
            print("Using CLIP with text_projection scrambled")
            ModelClass = ScrambledClip
        elif cfg.MODEL.CLIP.MISALIGNMENT == "mismatch":
            print("Using CLIP with a wrong text model")
            ModelClass = MisalignedClip
        else:
            ModelClass = ClipModel
        kwargs["pretrain"] = cfg.MODEL.IMAGE_MODEL.PRETRAINED
    elif cfg.MODEL.COMPOSITION in ["clipmaaf", "clipresmaaf"]:
        kwargs.update({"img_out_features": cfg.MODEL.IMAGE_MODEL.OUTPUTS,
                       "model_dim": cfg.MODEL.EMBED_DIM,
                       "num_heads": cfg.MODEL.MAAF.ATTENTION_HEADS,
                       "ff_width": cfg.MODEL.MAAF.BLOCK_WIDTH,
                       "dropout": cfg.MODEL.DROPOUT_RATE,
                       "num_blocks": cfg.MODEL.MAAF.NUM_BLOCKS,
                       "position_encodings": cfg.MODEL.MAAF.POSITION_ENCODING,
                       "output": cfg.MODEL.MAAF.OUTPUT,
                       "misalignment": cfg.MODEL.CLIP.MISALIGNMENT})
        if cfg.MODEL.COMPOSITION == "clipresmaaf":
            kwargs["learn_weight"] = cfg.MODEL.MAAF.RESIDUAL.LEARN_WEIGHTS
            if cfg.MODEL.MAAF.RESIDUAL.INITIAL_MAAF_PRESIGMOID is not None:
                kwargs["maaf_weight"] = torch.sigmoid(torch.Tensor([
                    cfg.MODEL.MAAF.RESIDUAL.INITIAL_MAAF_PRESIGMOID
                ])).item()
            else:
                kwargs["maaf_weight"] = \
                    cfg.MODEL.MAAF.RESIDUAL.INITIAL_MAAF_WEIGHT
            ModelClass = ClipResMAAF
        else:
            ModelClass = ClipMAAF
    else:
        raise ValueError(f"Invalid architecture {cfg.MODEL.COMPOSITION}")

    return ModelClass, kwargs
