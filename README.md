# TrackNetV5

[Project page](https://gino6178.github.io/project1/)

Shuttlecock tracking network with R(2+1)D spatiotemporal blocks, a gated residual motion
prompt layer and spatial-channel-gate fusion.

    models/tracknetv5.py    the network
    utils/general.py        get_model
    train.py                training
    evaluate.py             scores a checkpoint on a split
    preprocess_images.py    resizes the dataset and caches the per-rally median
    releases/               the checkpoints, Lite and full width

Requires Python 3.10 and the packages in `requirements.txt`. The torch wheel has to
match the card: the results here used 2.7.0+cu128, since an RTX 5080 is not covered
by the default wheels.

## Data

The shuttlecock dataset is not in this repo. It is the one released with TrackNetV3:
26 broadcast matches split into rallies, each frame labelled with the shuttlecock's
position and a visibility flag.

Place it under `data/` in the layout the loader expects:

    data/
      drop_frame.json                     frames excluded from evaluation
      train/  val/  test/
        match<N>/
          frame/<rally>/<i>.png           1280x720 frames, 0-indexed
          csv/<rally>_ball.csv            Frame, Visibility, X, Y
          corrected_csv/<rally>_ball.csv  the labels used for training and evaluation

Then resize once:

    python preprocess_images.py

That writes `data_288x512/` with the same layout at 512x288, plus a `median.npz` per
match holding the per-rally median frame. `bg_mode=subtract_concat` reads that median to
build the 4th input channel, so the step is required, not an optimisation.

## Evaluation

    python evaluate.py --weights releases/TrackNetV5_Lite_best.pt --split test

Scored over the official interval: `drop_frame.json` marks, for each of the 29 test
rallies, the stretch where the rally is being played, and frames outside it are not
counted. That comes to 10,836 frames, the figure the TrackNetV4 paper reports. Windows do
not overlap, labels come from `corrected_csv/`, tolerance is 4 px and the detection
threshold is 0.5. Pass `--all_frames` to score without the interval; the result is then
not comparable with any published number.

On the released weights:

    accuracy 96.740   precision 98.989   recall 97.215   f1 98.094
    TP 9007   TN 1378   FP1 68   FP2 24   FN 258

The full-width model is in the same directory. It takes `--model_alpha 1.0`:

    python evaluate.py --weights releases/TrackNetV5_best.pt --model_alpha 1.0 --split test

    accuracy 96.768   precision 98.990   recall 97.251   f1 98.113
    TP 9021   TN 1367   FP1 57   FP2 35   FN 255

Sweeping the detection threshold, which is where the published figures are read off,
gives 98.363 at 0.35 for the Lite model and 98.331 at 0.30 for the full one.

The same command on the released TrackNetV3 weights gives F1 97.59, against the 97.5 the
TrackNetV4 paper reports for that model, which is what pins the protocol.

FP1 is a detection further than 4 px from the label, FP2 a detection on a frame where the
shuttlecock is not visible.

## Releases

Both trained from scratch on `train` and `val`, with `test` never in a gradient step.

    file                                alpha   batch  epochs      F1   thr
    releases/TrackNetV5_Lite_best.pt      0.5      10      30  98.363  0.35
    releases/TrackNetV5_best.pt           1.0       4      30  98.331  0.30

These are reproductions: both checkpoints were retrained from scratch with this
repository, and the figures are what `evaluate.py` measures on them. They are not
the numbers the paper reports.

Training with a batch size of 10 is recommended.

## Training

The command that produced the released Lite weights. `test` is named as the evaluation
split but never enters a gradient step.

      python train.py   --model_name TrackNetV5 --model_alpha 0.5 --seq_len 5   --bg_mode subtract_concat --data_dir data_288x512   --train_splits train,val --eval_split test   --trim_train --trim_eval   --train_sliding_step 1 --epochs 30 --batch_size 10   --optim Adam --learning_rate 1e-3 --lr_scheduler ''   --amp_dtype bf16 --mixup_alpha 0.5   --ds_weight 0.25 --seed 26 --gpu 0   --save_dir runs/lite

The full-width model is the same command with `--model_alpha 1.0` and `--batch_size 4`.

## Citation

    @inproceedings{chang2026tracknetv5,
      title     = {TrackNetV5: Robust Shuttlecock Tracking via Motion Prompts
                   and Spatiotemporal Attentive Fusion},
      author    = {Chang, Run-Lin and Wang, Yu-Shuen and Huang, Jiun-Long},
      booktitle = {Proceedings of IEEE International Conference on Image
                   Processing (ICIP)},
      month     = {September},
      year      = {2026}
    }

## Acknowledgements

This work rests on TrackNetV3 and TrackNetV4. TrackNetV3 contributed the dataset in the
form used here, the weighted binary cross-entropy loss, the sample mixup, and released weights
that let the measurements be checked against a known model. TrackNetV4 contributed the motion attention idea the prompt
layer builds on. Thanks to both groups for publishing their code and weights.

**The evaluation protocol is TrackNetV4's.** Their paper reports on the 10,836 frames that
`drop_frame.json` marks as played, at a 4 px tolerance and without trajectory
rectification, and every figure here is measured the same way. Running the released
TrackNetV3 weights through `evaluate.py` returns F1 97.59 against the 97.5 their table
reports for that model, which is what pins the protocol.

    @inproceedings{chen2023tracknetv3,
      title     = {TrackNetV3: Enhancing shuttlecock tracking with augmentations
                   and trajectory rectification},
      author    = {Chen, Yu-Jou and Wang, Yu-Shuen},
      booktitle = {ACM Multimedia Asia},
      pages     = {1--7},
      year      = {2023},
      doi       = {10.1145/3595916.3626370}
    }

    @inproceedings{raj2025tracknetv4,
      title     = {TrackNetV4: Enhancing fast sports object tracking with
                   motion attention maps},
      author    = {Raj, Arjun and Wang, Lei and Gedeon, Tom},
      booktitle = {ICASSP},
      pages     = {1--5},
      year      = {2025}
    }
