"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import argparse
from pathlib import Path

from PIL import Image

import os
import json
import random
from tqdm import tqdm

import torch

import utils
from utils import refine, save_tensor_to_image

from dataset import HandwrittenMEGeneratorEvalDataset
from models import Generator

from sconf import Config
from train import setup_transforms
from torchvision import transforms

def eval_ckpt():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", help="path to config.yaml")
    parser.add_argument("--weight", help="path to weight to evaluate.pth")
    parser.add_argument("--result_dir", help="path to save the result file")
    args, left_argv = parser.parse_known_args()

    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml")
    cfg.argv_update(left_argv)
    img_dir = Path(args.result_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    val_transform = transforms.Compose([
        transforms.Pad(10, fill=255),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    specials = json.load(open("data/specials.json", 'r'))
    numbers = [str(c) for c in range(0, 10)]
    alphabets = [chr(c) for c in range(ord('a'), ord('z') + 1)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    etc = ["!", "'", "(", ")", "+", ",", "-", ".", "/", ";", "=", ">", "<", "[", "]", "|"]

    images = os.listdir("data/data_png_TrainingPrinted/")
    images = ["data/data_png_TrainingPrinted/" + image for image in images]

    chars = ["pad"] + numbers + alphabets + etc + list(specials.values())
    chars.sort()

    test_len = 64

    ttfs = ["data/ttfs/" + font_file for font_file in os.listdir("data/ttfs")]
    ttfs.sort()

    g_kwargs = cfg.get('g_args', {})
    gen = Generator(1, cfg.C, 1, **g_kwargs).cuda()

    weight = torch.load(args.weight)
    if "generator_ema" in weight:
        weight = weight["generator_ema"]
    gen.load_state_dict(weight)
    test_set = HandwrittenMEGeneratorEvalDataset(data=random.sample(images, test_len), chars=chars, ttfs=ttfs, transform=val_transform)
    test_loader = test_set.get_dataloader(batch_size=2, shuffle=False)

    for batch in tqdm(test_loader):
        style_imgs = batch["style_imgs"].cuda()
        char_imgs = batch["char_imgs"].cuda()

        predicts = []

        for style_img, char_img in zip(style_imgs, char_imgs):
            style_img = style_img.unsqueeze(1)
            char_img = char_img.unsqueeze(1)
            out = gen.gen_from_style_char(style_img, char_img)

            predicts.append(out)

        fonts = batch["fonts"]
        chars = batch["chars"]

        index = 0

        for image, font, char in zip(predicts, fonts, chars):
            font = font.split('/')[-1].split(".")[0]
            char = char.replace("/", "_")
            (img_dir / font).mkdir(parents=True, exist_ok=True)
            path = img_dir / font / f"{char}.png"

            if char_imgs[index][1].mean() == 1.:
                grid = utils.make_comparable_grid(char_imgs[index][0].unsqueeze(0), image[0].unsqueeze(0), nrow=1)
            else:
                grid = utils.make_comparable_grid(char_imgs[index], image, nrow=2)
            # grid = utils.make_comparable_grid(image.unsqueeze(0), nrow=1)
            save_tensor_to_image(grid, path)
            
            index += 1

if __name__ == "__main__":
    eval_ckpt()
