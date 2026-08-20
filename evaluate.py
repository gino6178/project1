"""Score a checkpoint on one split.

Windows do not overlap (sliding_step = seq_len), labels come from corrected_csv and the
frames listed in drop_frame.json are excluded, which is what eval_tracknet does and what
the training loop reports each epoch. Running this on the released weights should
reproduce the number in that run's log.

    python evaluate.py --weights releases/TrackNetV5_Lite_best.pt --split test
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import Shuttlecock_Trajectory_Dataset   # noqa: E402
from test import eval_tracknet                       # noqa: E402
from utils.general import get_model                  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='releases/TrackNetV5_Lite_best.pt')
    p.add_argument('--model_name', default='TrackNetV5')
    p.add_argument('--model_alpha', type=float, default=0.5)
    p.add_argument('--seq_len', type=int, default=5)
    p.add_argument('--bg_mode', default='subtract_concat')
    p.add_argument('--data_dir', default='data_288x512')
    p.add_argument('--split', default='test')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--tolerance', type=float, default=4.0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--all_frames', action='store_true',
                   help='score every frame instead of the official interval. The '
                        'published figures use the interval, so this is not comparable '
                        'with them')
    a = p.parse_args()

    dev = f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu'
    model = get_model(a.model_name, a.seq_len, a.bg_mode, alpha=a.model_alpha)
    ck = torch.load(a.weights, map_location='cpu', weights_only=False)
    sd = ck['model_state_dict'] if 'model_state_dict' in ck else ck
    model.load_state_dict(sd, strict=True)
    model = model.to(dev).eval()
    print(f'  {a.model_name}  alpha={a.model_alpha}  '
          f'{sum(q.numel() for q in model.parameters()):,} params  on {dev}')

    ds = Shuttlecock_Trajectory_Dataset(
        root_dir=a.data_dir, split=a.split, seq_len=a.seq_len,
        sliding_step=a.seq_len, data_mode='heatmap', bg_mode=a.bg_mode)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, drop_last=False,
                        num_workers=min(os.cpu_count(), 8), pin_memory=True)
    print(f'  split {a.split}: {len(ds)} clips, windows do not overlap')

    if not a.all_frames:
        # drop_frame.json marks, for each test rally, the stretch where the rally is
        # actually being played. Outside it the shuttlecock is held for the serve or
        # lying on the court, and the published protocol does not score those frames.
        import json
        cand = [os.path.join(a.data_dir, 'drop_frame.json'),
                os.path.join('data', 'drop_frame.json')]
        dfp = next((c for c in cand if os.path.exists(c)), None)
        if dfp is None:
            raise FileNotFoundError(
                'drop_frame.json not found in ' + ' or '.join(cand) +
                '. It ships with the dataset and defines the evaluation interval; '
                'pass --all_frames to score without it.')
        d = json.load(open(dfp))
        lo, hi = d['start'], d['end']
        ids = ds.data_dict['id']
        i2p = ds.rally_dict['i2p']
        keep = np.ones(len(ids), bool)
        for n in range(len(ids)):
            for ri, fi in ids[n]:
                path = i2p[int(ri)]
                key = (f"{path.split(os.sep)[-3].replace('match', '')}_"
                       f"{os.path.basename(path)}")
                if key in lo and not (int(lo[key]) <= int(fi) < int(hi[key])):
                    keep[n] = False
                    break
        before = len(ids)
        for k in list(ds.data_dict.keys()):
            ds.data_dict[k] = ds.data_dict[k][keep]
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, drop_last=False,
                            num_workers=min(os.cpu_count(), 8), pin_memory=True)
        print(f'  official interval: {before} -> {len(ds)} clips '
              f'({a.seq_len * len(ds):,} frames)')

    loss, r = eval_tracknet(model, loader, {'verbose': True, 'tolerance': a.tolerance})

    print(f'\n  loss       {loss:.6f}')
    print(f'  accuracy   {100*r["accuracy"]:.3f}')
    print(f'  precision  {100*r["precision"]:.3f}')
    print(f'  recall     {100*r["recall"]:.3f}')
    print(f'  f1         {100*r["f1"]:.3f}')
    print(f'  TP {r["TP"]:.0f}  TN {r["TN"]:.0f}  FP1 {r["FP1"]:.0f}  '
          f'FP2 {r["FP2"]:.0f}  FN {r["FN"]:.0f}')
    print(f'\n  FP1 is a detection further than {a.tolerance:.0f} px from the label, '
          f'FP2 a detection where the shuttlecock is not visible.')


if __name__ == '__main__':
    main()
