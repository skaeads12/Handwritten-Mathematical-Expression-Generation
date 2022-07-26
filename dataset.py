
import json
import math
import random
from tqdm import tqdm
from itertools import chain
from PIL import Image, ImageDraw, ImageFont

from torchvision.transforms import Compose

import torch
from torch.utils.data import Dataset, DataLoader

from render import render, render_matrix, render_frac

class HandwrittenMEGeneratorDataset(Dataset):

    def __init__(
        self,
        data: list = None,
        chars: list = None,
        ttfs: list = None,
        transform: Compose = None
    ):
        self.data = data
        self.n_samples = len(data)

        self.chars = chars
        self.n_chars = len(chars)

        self.ttfs = ttfs
        self.n_fonts = len(ttfs)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def get_dataloader(self, batch_size: int = None, use_ddp: bool = False):
        if use_ddp:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(self)
        else:
            sampler = None

        return DataLoader(self, batch_size=batch_size, collate_fn=self.collate_fn, sampler=sampler)

    def __getitem__(self, index):

        fonts = random.sample(self.ttfs, 4)

        target_font = fonts[0]
        ref_fonts = fonts[1:]

        sample = self.data[index]
        target_chars = sample["chars"]

        exp_flags = [random.random() > 0.9 for _ in target_chars]
        for i, exp_flag in enumerate(exp_flags):
            if i == 0:
                exp_flags[i] = False
                continue

            if exp_flag and exp_flags[i-1]:
                exp_flags[i] = False

        base_flags = [random.random() > 0.9 for _ in target_chars]
        for i, base_flag in enumerate(base_flags):
            if i == 0:
                base_flags[i] = False
                continue

            if base_flag and base_flags[i-1]:
                base_flags[i] = False

        crop_selector = random.random() > 0.5

        if sample["has_matrix"]:
            target_image = render_matrix(target_font, target_chars, exp_flags, base_flags, sample["matrix_type"])
        elif sample["has_frac"]:
            target_image = render_frac(target_font, target_chars, exp_flags, base_flags, sample["frac_size"])
        else:
            target_image = render(target_font, target_chars, exp_flags, base_flags)
        
        if len(target_image) == 2 and crop_selector:
            target_image = target_image[1]
        else:
            target_image = target_image[0]

        target_image = self.transform(target_image)

        style_images = []
        style_chars = []
        while len(style_images) < 3:
            ref_chars = random.sample(self.chars[1:], random.choice(range(1, 16)))
            if random.random() > 0.9:
                style_image = render_matrix(target_font, ref_chars, matrix_type=random.choice(["p", "v", "b", "n"]))[0]
            elif random.random() > 0.9:
                style_image = render_frac(target_font, ref_chars, frac_size=random.choice([2, 4, 8]))[0]
            else:
                style_image = render(target_font, ref_chars)[0]
            style_image = self.transform(style_image)
            style_images.append(style_image)
            style_chars.append([self.chars.index(char) for char in ref_chars] + [0] * (32 - len(ref_chars)))
        
        char_images = []

        for ref_font in ref_fonts:

            if sample["has_matrix"]:
                char_image = render_matrix(ref_font, target_chars, exp_flags, base_flags, sample["matrix_type"])
            elif sample["has_frac"]:
                char_image = render_frac(ref_font, target_chars, exp_flags, base_flags, sample["frac_size"])
            else:
                char_image = render(ref_font, target_chars, exp_flags, base_flags)

            if len(char_image) == 2 and crop_selector:
                char_image = char_image[1].resize((64, 64))
            else:
                char_image = char_image[0].resize((64, 64))

            char_images.append(self.transform(char_image))

        target_chars = [self.chars.index(char) for char in target_chars]
        if len(target_chars) < 32:
            target_chars += [0] * (32 - len(target_chars))
        else:
            target_chars = target_chars[:32]

        target_fids = [self.ttfs.index(target_font)]
        char_fids = [self.ttfs.index(ref_font) for ref_font in ref_fonts]

        return {
            "style_imgs": torch.stack(style_images),
            "style_fids": torch.LongTensor(target_fids * 3),
            "style_decs": style_chars,
            "char_imgs": torch.stack(char_images),
            "char_fids": torch.LongTensor(char_fids),
            "char_decs": [target_chars] * 3,
            "trg_imgs": target_image,
            "trg_fids": torch.LongTensor(target_fids),
            "trg_cids": torch.LongTensor(target_chars),
            "trg_decs": target_chars,
        }

    @staticmethod
    def collate_fn(batch):
        _ret = {}
        for dp in batch:
            for key, value in dp.items():
                saved = _ret.get(key, [])
                _ret.update({key: saved + [value]})

        ret = {
            "trg_imgs": torch.stack(_ret["trg_imgs"]),
            "trg_decs": _ret["trg_decs"],
            "trg_fids": torch.cat(_ret["trg_fids"]),
            "trg_cids": torch.stack(_ret["trg_cids"]),
            "style_imgs": torch.stack(_ret["style_imgs"]),
            "style_decs": [*chain(*_ret["style_decs"])],
            "style_fids": torch.stack(_ret["style_fids"]),
            "char_imgs": torch.stack(_ret["char_imgs"]),
            "char_decs": [*chain(*_ret["char_decs"])],
            "char_fids": torch.stack(_ret["char_fids"])
        }

        return ret

class HandwrittenMEGeneratorEvalDataset(Dataset):

    def __init__(
        self,
        data: list = None,
        chars: list = None,
        ttfs: list = None,
        transform: Compose = None

    ):
        self.data = data
        self.n_samples = len(self.data)

        self.chars = chars
        self.n_chars = len(self.chars)

        self.ttfs = ttfs
        self.n_fonts = len(self.ttfs)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def get_dataloader(self, batch_size: int = None, shuffle: bool = False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def __getitem__(self, index):

        image_file = self.data[index]

        char_image = Image.open(image_file).convert("L")
        width, height = char_image.size

        if width > height * 8:
            char_image1 = char_image.crop((0, 0, width // 2, height))
            char_image1 = self.transform(char_image1)
            char_image2 = char_image.crop((width // 2, 0, width, height))
            char_image2 = self.transform(char_image2)
        elif height > width * 5:
            char_image1 = char_image.crop((0, 0, width, height // 2))
            char_image1 = self.transform(char_image1)
            char_image2 = char_image.crop((0, height // 2, width, height))
            char_image2 = self.transform(char_image2)
        else:
            char_image1 = self.transform(char_image)
            char_image2 = self.transform(Image.new("L", size=(256, 256), color=255))

        char_images = [char_image1, char_image2]

        font = random.choice(self.ttfs)
        ref_chars = [random.sample(self.chars, random.choice(range(1, 16))) for _ in char_images]

        style_images = [self.transform(render(font, ref_char)[0]) for ref_char in ref_chars]

        return {
            "style_imgs": torch.stack(style_images),
            "char_imgs": torch.stack(char_images),
            "fonts": font,
            "chars": image_file.split("/")[-1].split(".")[0],
        }