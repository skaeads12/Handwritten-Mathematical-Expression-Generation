"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import json
import os
import re
import sys
from pathlib import Path
import argparse
import random

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torchvision import transforms
import numpy as np
from sconf import Config, dump_args
import utils
from utils import Logger

from models import Generator, disc_builder, aux_clf_builder
from models.modules import weights_init
from trainer import FactTrainer, Evaluator, load_checkpoint
from dataset import HandwrittenMEGeneratorDataset

def setup_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", help="path/to/config.yaml")

    args, left_argv = parser.parse_known_args()

    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml",
                 colorize_modified_item=True)
    cfg.argv_update(left_argv)

    if cfg.use_ddp:
        cfg.n_workers = 0

    cfg.work_dir = Path(cfg.work_dir)
    (cfg.work_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    return args, cfg


def setup_transforms(cfg):
    if cfg.dset_aug.random_affine:
        aug_transform = [
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=10, translate=(0.03, 0.03), scale=(0.9, 1.1), shear=10, fillcolor=255
            ),
        ]
    else:
        aug_transform = []

    tensorize_transform = [transforms.Resize((256, 256)), transforms.ToTensor()]
    if cfg.dset_aug.normalize:
        tensorize_transform.append(transforms.Normalize([0.5], [0.5]))
        cfg.g_args.dec.out = "tanh"

    trn_transform = transforms.Compose(aug_transform + tensorize_transform)
    val_transform = transforms.Compose(tensorize_transform)

    return trn_transform, val_transform


def cleanup():
    dist.destroy_process_group()


def is_main_worker(gpu):
    return (gpu <= 0)


def train_ddp(gpu, args, cfg, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:" + str(cfg.port),
        world_size=world_size,
        rank=gpu,
    )
    cfg.batch_size = cfg.batch_size // world_size
    train(args, cfg, ddp_gpu=gpu)
    cleanup()


def train(args, cfg, ddp_gpu=-1):
    cfg.gpu = ddp_gpu
    torch.cuda.set_device(ddp_gpu)
    cudnn.benchmark = True

    logger_path = cfg.work_dir / "log.log"
    logger = Logger.get(file_path=logger_path, level="info", colorize=True)

    image_scale = 0.5
    image_path = cfg.work_dir / "images"
    writer = utils.DiskWriter(image_path, scale=image_scale)
    cfg.tb_freq = -1

    args_str = dump_args(args)
    if is_main_worker(ddp_gpu):
        logger.info("Run Argv:\n> {}".format(" ".join(sys.argv)))
        logger.info("Args:\n{}".format(args_str))
        logger.info("Configs:\n{}".format(cfg.dumps()))

    logger.info("Get dataset ...")

    trn_transform, val_transform = setup_transforms(cfg)

    specials = json.load(open("data/specials.json", 'r'))
    numbers = [str(c) for c in range(0, 10)]
    alphabets = [chr(c) for c in range(ord('a'), ord('z') + 1)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    etc = ["!", "'", "(", ")", "+", ",", "-", ".", "/", ";", "=", ">", "<", "[", "]", "|"]

    total_labels = json.load(open("data/labels.json", 'r'))
    total_set = []

    for label in total_labels.values():

        has_matrix = False
        if "matrix" in label:
            has_matrix = True

        matrix_type = "n"

        has_frac = False
        if "frac" in label:
            has_frac = True

        chars = []
        for token in label.split(" "):
            if token.startswith("\\") and token[1:] in specials:
                token = specials[token[1:]]
            
            if token in specials.values() or token in alphabets or token in numbers or token in etc:
                chars.append(token)

            if token == "pmatrix":
                matrix_type = "p"
            elif token.lower() == "vmatrix":
                matrix_type = "v"
            elif token == "bmatrix":
                matrix_type = "b"

        frac_size = 2

        if has_frac:
            char_len = len(chars)

            while (char_len // frac_size) > 2:
                if random.random() > 0.5:
                    frac_size *= 2

                char_len = char_len // 2

        total_set.append({
            "chars": chars,
            "has_matrix": has_matrix,
            "matrix_type": matrix_type,
            "has_frac": has_frac,
            "frac_size": frac_size
        })

    chars = ["pad"] + numbers + alphabets + etc + list(specials.values())
    chars.sort()

    total_len = len(total_set)
    train_len = int(total_len * 0.9)
    random.shuffle(total_set)

    ttfs = ["data/ttfs/" + font_file for font_file in os.listdir("data/ttfs")]
    ttfs.sort()

    train_set = HandwrittenMEGeneratorDataset(data=total_set[:train_len], chars=chars, ttfs=ttfs, transform=trn_transform)
    train_loader = train_set.get_dataloader(batch_size=cfg.batch_size, use_ddp=cfg.use_ddp)

    test_set = HandwrittenMEGeneratorDataset(data=total_set[train_len:], chars=chars, ttfs=ttfs, transform=val_transform)
    test_loader = test_set.get_dataloader(batch_size=cfg.batch_size, use_ddp=False)

    primals = train_set.chars
    n_comps = len(primals)

    logger.info("Build model ...")
    # generator
    g_kwargs = cfg.get("g_args", {})
    gen = Generator(1, cfg.C, 1, **g_kwargs)
    gen.cuda()
    gen.apply(weights_init(cfg.init))

    d_kwargs = cfg.get("d_args", {})
    disc = disc_builder(cfg.C, train_set.n_fonts, train_set.n_chars, **d_kwargs)
    disc.cuda()
    disc.apply(weights_init(cfg.init))

    aux_clf = aux_clf_builder(gen.feat_shape["last"], train_set.n_fonts, n_comps, **cfg.ac_args)
    aux_clf.cuda()
    aux_clf.apply(weights_init(cfg.init))

    g_optim = optim.Adam(gen.parameters(), lr=cfg.g_lr, betas=cfg.adam_betas)
    d_optim = optim.Adam(disc.parameters(), lr=cfg.d_lr, betas=cfg.adam_betas)
    ac_optim = optim.Adam(aux_clf.parameters(), lr=cfg.ac_lr, betas=cfg.adam_betas)

    # use pretrained model
    weight = torch.load("result/checkpoints/last.pth")
    if "generator_ema" in weight:
        gen.load_state_dict(weight["generator_ema"])
        g_optim.load_state_dict(weight["optimizer"])
        print("generator loaded.")
    
    if "discriminator" in weight:
        disc.load_state_dict(weight["discriminator"])
        d_optim.load_state_dict(weight["d_optimizer"])
        print("discriminator loaded.")
    
    if "aux_clf" in weight:
        aux_clf.load_state_dict(weight["aux_clf"])
        ac_optim.load_state_dict(weight["ac_optimizer"])
        print("aux clf loaded.")

    st_step = 0
    if cfg.resume:
        st_step, loss = load_checkpoint(cfg.resume, gen, disc, aux_clf, g_optim, d_optim, ac_optim, cfg.force_resume)
        logger.info("Resumed checkpoint from {} (Step {}, Loss {:7.3f})".format(cfg.resume, st_step, loss))

    evaluator = Evaluator(writer)

    trainer = FactTrainer(gen, disc, g_optim, d_optim,
                          aux_clf, ac_optim,
                          writer, logger,
                          evaluator, test_loader,
                          cfg)

    trainer.train(train_loader, st_step, cfg.max_iter)


def main():
    args, cfg = setup_args_and_config()

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    if cfg.use_ddp:
        ngpus_per_node = torch.cuda.device_count()
        world_size = ngpus_per_node
        mp.spawn(train_ddp, nprocs=ngpus_per_node, args=(args, cfg, world_size))
    else:
        train(args, cfg)


if __name__ == "__main__":
    main()
