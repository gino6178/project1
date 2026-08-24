"""Parametrised trainer for sweeping the V5x variants.

Same logic as trainv4.py (their tuned recipe: Adadelta 1.0 + ReduceLROnPlateau, AMP,
cudnn.benchmark, eval via eval_tracknet so corrected_csv + drop_frame apply). The only
change is that every setting is a CLI argument, and per-epoch metrics are appended to a
CSV so runs can be compared without parsing logs.
"""
import argparse
import csv
import os
import time

import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, default='TrackNetV5')
    p.add_argument('--model_alpha', type=float, default=0.5)
    p.add_argument('--seq_len', type=int, default=3)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--optim', type=str, default='Adadelta',
                   choices=['Adadelta', 'Adam', 'AdamW', 'SGD'])
    p.add_argument('--learning_rate', type=float, default=1.0)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau',
                   choices=['', 'ReduceLROnPlateau', 'CosineAnnealingLR'])
    p.add_argument('--bg_mode', type=str, default='subtract_concat',
                   choices=['', 'subtract', 'subtract_concat', 'concat'])
    p.add_argument('--seed', type=int, default=26)
    p.add_argument('--data_dir', type=str, default='data_288x512')
    p.add_argument('--train_sliding_step', type=int, default=1,
                   help='1 reproduces their recipe; larger values subsample uniformly')
    p.add_argument('--data_ratio', type=float, default=1.0)
    p.add_argument('--save_dir', type=str, required=True)
    p.add_argument('--tag', type=str, default='')
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--tolerance', type=float, default=4.0)
    p.add_argument('--ds_weight', type=float, default=0.25,
                   help='weight on each auxiliary head loss (the paper does not\n                         state it; 0.25 is the common choice)')
    p.add_argument('--train_splits', type=str, default='train',
                   help='comma separated splits to train on, e.g. train,val')
    p.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16', 'fp16'],
                   help='autocast precision. fp16 overflows in up_block_3 on this '
                        'network; bf16 has the exponent range of fp32')
    p.add_argument('--mixup_alpha', type=float, default=-1.0,
                   help='beta parameter for sample mixup; -1 disables it. The released '
                        'TrackNetV3 used 0.5')
    p.add_argument('--trim_train', action='store_true',
                   help='train only inside each rally\'s play interval')
    p.add_argument('--trim_eval', action='store_true',
                   help='evaluate only inside it, which is the official protocol')
    p.add_argument('--trim_speed', type=float, default=3.0,
                   help='px/frame that counts as the shuttlecock being in play')
    p.add_argument('--teacher_weights', type=str, default='',
                   help='checkpoint to distil from; empty disables distillation')
    p.add_argument('--teacher_name', type=str, default='',
                   help='teacher model name, defaults to --model_name')
    p.add_argument('--kd_weight', type=float, default=0.0,
                   help='weight of the soft-target term')
    p.add_argument('--eval_split', type=str, default='val',
                   help='split to evaluate each epoch; the paper and the official\n                         V3 recipe both use test')
    p.add_argument('--init_from', type=str, default='',
                   help='checkpoint to load model weights from; optimiser state is\n                         NOT loaded, so the schedule starts fresh')
    p.add_argument('--quiet', action='store_true')
    return p.parse_args()


ARGS = get_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.gpu

import torch                                                # noqa: E402
from torch.utils.data import DataLoader                      # noqa: E402
from torch.amp import autocast, GradScaler                   # noqa: E402
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR  # noqa: E402
from tqdm import tqdm                                        # noqa: E402

from dataset import Shuttlecock_Trajectory_Dataset           # noqa: E402
from test import eval_tracknet                               # noqa: E402
from utils.general import get_model                          # noqa: E402


def mixup(x, y, alpha=0.5):
    """Blend pairs within the batch, targets included, as TrackNetV3 does."""
    n = x.size()[0]
    lamb = np.random.beta(alpha, alpha, size=n)
    lamb = np.maximum(lamb, 1 - lamb)
    lamb = torch.from_numpy(lamb[:, None, None, None]).float().to(x.device)
    idx = torch.randperm(n)
    return x * lamb + x[idx] * (1 - lamb), y * lamb + y[idx] * (1 - lamb)

from utils.metric import WBCELoss                            # noqa: E402


AMP_DTYPE = torch.bfloat16 if ARGS.amp_dtype == 'bf16' else torch.float16


def train_one_epoch(model, optimizer, loader, scaler, verbose, ds_w=0.0,
                    teacher=None, kd_w=0.0, mix_a=-1.0):
    model.train()
    losses = []
    it = tqdm(loader, mininterval=10.0) if verbose else loader
    for step, (_, x, y, _, _) in enumerate(it):
        optimizer.zero_grad(set_to_none=True)
        x = x.float().cuda(non_blocking=True)
        y = y.float().cuda(non_blocking=True)
        if mix_a > 0:
            # blends pairs within the batch, targets included, as the released
            # TrackNetV3 was trained
            x, y = mixup(x, y, mix_a)
        with autocast(device_type='cuda', dtype=AMP_DTYPE):
            out = model(x)
            if isinstance(out, (tuple, list)):
                main, aux = out
                loss = WBCELoss(main, y) + ds_w * sum(WBCELoss(a, y) for a in aux)
            else:
                main = out
                loss = WBCELoss(out, y)
            if teacher is not None and kd_w > 0:
                # the teacher's heatmap as a soft target, through the same weighted BCE
                # the hard labels use, so the two terms are on one scale
                with torch.no_grad():
                    t = teacher(x)
                    t = t[0] if isinstance(t, (tuple, list)) else t
                loss = loss + kd_w * WBCELoss(main, t.detach().clamp(0, 1))
        if not torch.isfinite(loss):
            raise RuntimeError(
                f'NON-FINITE LOSS at step {step}: {float(loss)}. Training aborted; '
                f'continuing would only propagate it through every later epoch.')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        if verbose and (step + 1) % 200 == 0:
            it.set_postfix(loss=loss.item())
    return float(np.mean(losses))



def _rally_key(path):
    import os
    parts = path.split(os.sep)
    return f"{parts[-3].replace('match', '')}_{os.path.basename(path)}"


def _intervals(root, split, speed_thr=3.0):
    """[start, end) per rally key.

    The released drop_frame.json covers the test rallies only. Everywhere else the same
    boundary is recovered from the labels: the first and last frame where the shuttlecock
    is moving. Checked against the annotation on the 29 rallies that have one, the start
    is off by 2.2 frames on average and the end by 5.8.
    """
    import glob, json, os
    import pandas as pd
    exact = {}
    # the released annotation describes test rallies; train and val reuse the same
    # match/rally names, so applying it outside test would attach the wrong interval
    if split == 'test':
        dfp = os.path.join(root, 'drop_frame.json')
        if not os.path.exists(dfp):
            dfp = os.path.join(os.path.dirname(root.rstrip('/')), 'data',
                               'drop_frame.json')
        if os.path.exists(dfp):
            d = json.load(open(dfp))
            exact = {k: (int(d['start'][k]), int(d['end'][k])) for k in d['map']}

    out = {}
    for rally in sorted(glob.glob(os.path.join(root, split, 'match*', 'frame', '*'))):
        key = _rally_key(rally)
        if key in exact:
            out[key] = exact[key]
            continue
        base = os.path.dirname(os.path.dirname(rally))
        rid = os.path.basename(rally)
        csv = os.path.join(base, 'corrected_csv', f'{rid}_ball.csv')
        if not os.path.exists(csv):
            csv = os.path.join(base, 'csv', f'{rid}_ball.csv')   # train/val have only this
        if not os.path.exists(csv):
            continue
        lab = pd.read_csv(csv)
        v = lab['Visibility'].to_numpy()
        x = lab['X'].to_numpy() / (1280 / 512)
        y = lab['Y'].to_numpy() / (720 / 288)
        moving = []
        for i in range(1, len(v)):
            if v[i] > 0 and v[i - 1] > 0 and \
                    np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) > speed_thr:
                moving.append(i)
        if moving:
            out[key] = (moving[0], moving[-1])
        else:
            vis = np.where(v > 0)[0]
            out[key] = (int(vis[0]), int(vis[-1])) if len(vis) else (0, len(v))
    return out


def _restrict(ds, iv, label):
    """drop every window that steps outside its rally's interval"""
    ids = ds.data_dict['id']                      # (N, L, 2) of (rally, frame)
    i2p = ds.rally_dict['i2p']
    n_rally = max(i2p.keys()) + 1
    lo = np.zeros(n_rally, np.int64)
    hi = np.full(n_rally, np.iinfo(np.int64).max)
    for ri, path in i2p.items():
        k = _rally_key(path)
        if k in iv:
            lo[ri], hi[ri] = iv[k]
    r, f = ids[..., 0], ids[..., 1]
    keep = ((f >= lo[r]) & (f < hi[r])).all(axis=1)
    before = len(ids)
    for k in list(ds.data_dict.keys()):
        ds.data_dict[k] = ds.data_dict[k][keep]
    print(f'  trimmed {label}: {before} -> {int(keep.sum())} clips '
          f'({100 * (1 - keep.mean()):.1f}% dropped)', flush=True)
    return ds


def main():
    a = ARGS
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

    os.makedirs(a.save_dir, exist_ok=True)
    csv_path = os.path.join(a.save_dir, 'metrics.csv')
    print(f'=== {a.tag or a.model_name} ===', flush=True)
    print('  ' + '  '.join(f'{k}={v}' for k, v in sorted(vars(a).items())), flush=True)

    splits = [x.strip() for x in a.train_splits.split(',') if x.strip()]
    parts = [Shuttlecock_Trajectory_Dataset(
                 root_dir=a.data_dir, split=sp, seq_len=a.seq_len,
                 sliding_step=a.train_sliding_step, data_mode='heatmap',
                 bg_mode=a.bg_mode) for sp in splits]
    for sp, ds in zip(splits, parts):
        print(f'  train split {sp}: {len(ds)} clips', flush=True)
        if a.trim_train:
            _restrict(ds, _intervals(a.data_dir, sp, a.trim_speed), f'train {sp}')
    train_set = parts[0] if len(parts) == 1 else torch.utils.data.ConcatDataset(parts)
    if a.eval_split in splits:
        print(f'  WARNING eval_split={a.eval_split} is also being trained on -- the'
              f' reported metrics are NOT held out', flush=True)
    val_set = Shuttlecock_Trajectory_Dataset(
        root_dir=a.data_dir, split=a.eval_split, seq_len=a.seq_len,
        sliding_step=a.seq_len, data_mode='heatmap', bg_mode=a.bg_mode)
    if a.data_ratio < 1.0:
        n = int(len(train_set) * a.data_ratio)
        idx = np.random.choice(len(train_set), n, replace=False)
        train_set = torch.utils.data.Subset(train_set, idx)
    if a.trim_eval:
        _restrict(val_set, _intervals(a.data_dir, a.eval_split, a.trim_speed),
                  f'eval {a.eval_split}')
    print(f'  train clips {len(train_set)} (from {splits})   eval clips {len(val_set)} (split {a.eval_split})', flush=True)

    nw = min(os.cpu_count(), 8)
    train_loader = DataLoader(train_set, batch_size=a.batch_size, shuffle=True,
                              num_workers=nw, drop_last=True, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=a.batch_size, shuffle=False,
                            num_workers=nw, drop_last=False, pin_memory=True,
                            persistent_workers=True)

    model = get_model(a.model_name, a.seq_len, a.bg_mode, alpha=a.model_alpha).cuda()
    if a.init_from:
        ck = torch.load(a.init_from, map_location='cpu', weights_only=False)
        sd = ck.get('model_state_dict', ck)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'  initialised from {a.init_from}', flush=True)
        print(f'    checkpoint epoch {ck.get("epoch")}  val_f1 {ck.get("val_f1")}',
              flush=True)
        print(f'    missing keys {len(missing)}  unexpected {len(unexpected)}',
              flush=True)
        if missing or unexpected:
            print(f'    WARNING mismatch: missing={missing[:5]} '
                  f'unexpected={unexpected[:5]}', flush=True)

    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  {a.model_name} alpha={a.model_alpha}  params {n_par:,} '
          f'({n_par/1e6:.3f}M)', flush=True)

    if a.optim == 'Adadelta':
        opt = torch.optim.Adadelta(model.parameters(), lr=a.learning_rate,
                                   weight_decay=a.weight_decay)
    elif a.optim == 'AdamW':
        opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate,
                                weight_decay=a.weight_decay)
    elif a.optim == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=a.learning_rate, momentum=0.9,
                              weight_decay=a.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=a.learning_rate,
                               weight_decay=a.weight_decay)

    if a.lr_scheduler == 'ReduceLROnPlateau':
        sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    elif a.lr_scheduler == 'CosineAnnealingLR':
        sched = CosineAnnealingLR(opt, T_max=a.epochs)
    else:
        sched = None

    teacher = None
    if a.teacher_weights:
        teacher = get_model(a.teacher_name or a.model_name, a.seq_len, a.bg_mode,
                            alpha=a.model_alpha)
        tsd = torch.load(a.teacher_weights, map_location='cpu',
                         weights_only=False)['model_state_dict']
        teacher.load_state_dict(tsd, strict=True)
        teacher = teacher.cuda().eval()
        for q in teacher.parameters():
            q.requires_grad_(False)
        print(f'  teacher {a.teacher_name or a.model_name} from {a.teacher_weights}, '
              f'kd_weight {a.kd_weight}', flush=True)

    # a scaler only matters for fp16 gradient underflow; bf16 does not need one
    scaler = GradScaler(enabled=(a.amp_dtype == 'fp16'))
    best_f1, best_ep = 0.0, -1
    t_start = time.time()

    for ep in range(a.epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(model, opt, train_loader, scaler, not a.quiet,
                                  ds_w=a.ds_weight, teacher=teacher, kd_w=a.kd_weight,
                                  mix_a=a.mixup_alpha)
        val_loss, r = eval_tracknet(model, val_loader,
                                    {'verbose': False, 'tolerance': a.tolerance})
        if sched is not None:
            sched.step(val_loss) if a.lr_scheduler == 'ReduceLROnPlateau' else sched.step()

        row = dict(epoch=ep + 1, train_loss=tr_loss, val_loss=val_loss,
                   accuracy=r['accuracy'], precision=r['precision'],
                   recall=r['recall'], f1=r['f1'],
                   TP=r['TP'], TN=r['TN'], FP1=r['FP1'], FP2=r['FP2'], FN=r['FN'],
                   lr=opt.param_groups[0]['lr'], minutes=(time.time() - t0) / 60)
        new = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

        print(f'  ep {ep+1:>2}/{a.epochs}  train {tr_loss:.6f}  val {val_loss:.6f}  '
              f'acc {100*r["accuracy"]:.3f}  prec {100*r["precision"]:.3f}  '
              f'rec {100*r["recall"]:.3f}  f1 {100*r["f1"]:.3f}  '
              f'TP {r["TP"]} FN {r["FN"]} FP1 {r["FP1"]} FP2 {r["FP2"]}  '
              f'lr {opt.param_groups[0]["lr"]:.4g}  {row["minutes"]:.1f}min', flush=True)

        if r['f1'] > best_f1:
            best_f1, best_ep = r['f1'], ep + 1
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                        'val_f1': r['f1'], 'args': vars(a)},
                       os.path.join(a.save_dir, f'{a.model_name}_best.pt'))

    print(f'  DONE  best f1 {100*best_f1:.3f} at epoch {best_ep}  '
          f'total {(time.time()-t_start)/3600:.2f} h', flush=True)


if __name__ == '__main__':
    main()
