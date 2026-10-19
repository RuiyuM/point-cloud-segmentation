import argparse
import copy
import datetime
import glob
import yaml, argparse, copy, os, sys
from pathlib import Path
from prettytable import PrettyTable
import time
import tqdm
import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.cuda import amp
import torchsparse
from torchsparse import SparseTensor
import torchsparse.nn
import torchsparse.nn.functional
from annotator.data import build_dataloader
from annotator.model import build_network, load_data_to_gpu
from annotator.optim import build_optimizer, build_scheduler
from tools.utils.common import common_utils, commu_utils
from tools.utils.train.config import cfgs, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from tools.utils.train_utils import model_state_to_cpu
from annotator.active.sampler import RandomSelect, EntropySelect, MarginSelect, VCDSelect, Entropy_Greedy_Select, VCDSelect_statistic, Logit_Select_Reverse, Features_Entropy_Greedy_Select_Logit_Mean, No_stage_3_Features_Entropy_Greedy_Select_Logit_Mean, No_stage_1_Features_Entropy_Greedy_Select_Logit_Mean
import setproctitle
import wandb
import pickle
from torch.func import functional_call, vmap, grad
from tools.utils.common.lovasz_losses import lovasz_softmax
from torch.nn import CrossEntropyLoss
# -------------------------------------------


def build_lut(yaml_file: str | None, key: str):
    """
    Build a numpy LUT from YAML.

    key='learning_map_inv'  → original SemanticPOSS IDs
    key='learning_map'      → display IDs used by the viewer
    Returns None if yaml_file is None or key missing.
    """
    if yaml_file is None:
        return None
    y = yaml.safe_load(open(yaml_file))
    if key not in y:
        return None
    table = y[key]
    lut = np.zeros(max(table.keys()) + 1, dtype=np.uint32)
    lut[list(table.keys())] = list(table.values())
    return lut


@torch.no_grad()
def export_predictions(cfg_file, checkpoint, save_root,
                       split, batch_size, workers, map_yaml):

    # ---------- config & dataloader ------------------------------------
    cfg_from_yaml_file(cfg_file, cfgs)
    cfgs.DATA.SPLIT = split

    _, loader, _ = build_dataloader(
        data_cfgs   = cfgs.DATA,
        modality    = cfgs.MODALITY,
        batch_size  = batch_size or cfgs.OPTIM.BATCH_SIZE_PER_GPU,
        dist        = False,
        workers     = workers,
        logger      = None,
        training    = False,
        merge_all_iters_to_one_epoch = False,
        total_epochs = 1,
    )

    # ---------- network -------------------------------------------------
    ckpt  = torch.load(checkpoint, map_location='cpu')
    model = build_network(cfgs.MODEL, num_class=cfgs.DATA.NUM_CLASSES + 1)
    model.load_params(ckpt['model_state'], strict=True)
    model.cuda().eval()

    # ---------- output dir & LUTs --------------------------------------
    save_root = Path(save_root)
    (save_root / 'sequences').mkdir(parents=True, exist_ok=True)

    raw_lut  = build_lut(map_yaml, key='learning_map_inv')   # 13‑cls → original
    disp_lut = build_lut(map_yaml, key='learning_map')       # original → display
    if raw_lut is None or disp_lut is None:
        raise RuntimeError("map_yaml must contain both learning_map_inv and learning_map")

    # helper functions
    raw_map  = lambda a: raw_lut[a]
    disp_map = lambda a: disp_lut[a]

    # ---------- evaluation‑style loop ----------------------------------
    pbar = tqdm.tqdm(total=len(loader), desc='eval', dynamic_ncols=True)
    loader_iter = iter(loader)

    for _ in range(len(loader)):
        batch = next(loader_iter)
        load_data_to_gpu(batch)

        ret = model(batch)
        voxel_hash = torchsparse.nn.functional.sphash(batch['targets'].C)
        point_hash = torchsparse.nn.functional.sphash(batch['targets_mapped'].C)
        inv_map    = torchsparse.nn.functional.sphashquery(point_hash, voxel_hash)

        logits = ret['voxel_predict_logits'][inv_map]
        preds  = logits.argmax(1).cpu().numpy()                   # (total_pts,)

        # ---- split preds by scan in batch ----------------------------
        sample_ids = batch['targets_mapped'].C[:, 3].cpu().numpy()
        names = batch['name']

        for i, scan_name in enumerate(names):
            mask = sample_ids == i
            pred_scan = preds[mask]                               # (pts_in_scan,)

            p = Path(scan_name)                                   # …/sequences/03/velodyne/000001.bin
            frame = p.stem
            try:
                seq = p.parts[p.parts.index('sequences') + 1]
            except ValueError:
                seq = p.parts[-3]

            out_dir = save_root / 'sequences' / seq / 'predictions'
            out_dir.mkdir(parents=True, exist_ok=True)

            # 1) 13‑class → original IDs   2) original → viewer display IDs
            disp_ids = disp_map(raw_map(pred_scan)).astype(np.uint32)
            disp_ids.tofile(out_dir / f'{frame}.label')

        pbar.update()

    pbar.close()
    print(f'✅  Predictions saved under {save_root}')


# ----------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Dump .label files for Semantic‑POSS')
    ap.add_argument('--cfg_file', required=True)
    ap.add_argument('--ckp',      required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--workers',    type=int, default=4)
    ap.add_argument('--map_yaml',   required=True,
                    help='YAML that contains BOTH learning_map_inv and learning_map')
    args = ap.parse_args()

    export_predictions(
        args.cfg_file, args.ckp, args.save_dir, args.split,
        args.batch_size, args.workers, args.map_yaml)
