# Region Aware Blind Face Restoration

---

## TODO

1. [x] Dataset preparation pipeline (`src/preprocess/`).
2. [x] Evaluation pipeline (`src/metrics/`).
3. [ ] Checkpoints uploaded on Google Drive.
6. [ ] Example images and qualitative result figures.
4. [x] Demo — run from each baseline's official repo with the downloaded checkpoints.
5. [x] Inference and training — use the official repos linked above.

---

## 1. Setup (conda)

```bash
conda create -n rabfr python=3.10 -y
conda activate rabfr

pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0
pip install -r requirements.txt
```

For the IDS metric, download the ArcFace model and place it under `checkpoints/`:
https://drive.google.com/drive/folders/1k3RCSliF6PsujCMIdCD1hNM63EozlDIZ

---

## 2. Checkpoints

Fine-tuned weights for the four baselines:

**https://drive.google.com/drive/folders/1ZsEfyivmqn9ZEwPqPWW4UjSzWFeB9W0r?usp=share_link**

Download the four subfolders (`gfpgan`, `psfrgan`, `codeformer`, `difface`) and place them under `checkpoints/`.

---

## 3. Demo

Download the checkpoints from step 2 and run the demo provided in each baseline's official repository, pointing it at the corresponding checkpoint under `checkpoints/`.

- GFPGAN — https://github.com/TencentARC/GFPGAN
- PSFRGAN — https://github.com/chaofengc/PSFRGAN
- CodeFormer — https://github.com/sczhou/CodeFormer
- DifFace — https://github.com/zsyOAOA/DifFace

---

## 4. Dataset Preparation

Build the cleaned dataset from raw videos.

```bash
python src/preprocess/preprocess.py \
  --input data/raw_videos \
  --output data/saudi_dataset \
  --arcface checkpoints/resnet18_110.pth
```

Pipeline (one frame per second → MTCNN detect/align → blur+NaN filter → yaw>45° filter → ArcFace dedup, cap 5/identity → identity-disjoint train/test split → GFPGAN-style degradation → HQ/LQ pairs). Each stage is also runnable on its own (`extract_frames.py`, `detect_align.py`, `filter_damaged.py`, `filter_pose.py`, `filter_identity.py`, `split.py`, `degrade.py`).

Final dataset layout:

```
data/saudi_dataset/
├── splits/
│   ├── train.txt
│   └── test.txt
├── train/
│   ├── HQ/        clean 512×512 ground truth
│   └── LQ/        synthetically degraded counterpart (paired by filename)
└── test/
    ├── HQ/
    └── LQ/
```

---

## 5. Inference and Training

Both inference and training are done in each baseline's official repository. Clone the relevant one and follow its instructions — use the cleaned dataset from step 4 for training, and the fine-tuned checkpoints from step 2 for inference. Save inference outputs to a folder, then point `evaluate.py` at it (step 6).

- GFPGAN — https://github.com/TencentARC/GFPGAN
- PSFRGAN — https://github.com/chaofengc/PSFRGAN
- CodeFormer — https://github.com/sczhou/CodeFormer
- DifFace — https://github.com/zsyOAOA/DifFace

---

## 6. Evaluation

Folder-based: point the script at the ground-truth folder and the restored-images folder.

```bash
python src/metrics/evaluate.py \
  --dirGT path/to/ground_truth/ \
  --dirEI path/to/restored/ \
  --imagesize 512
```

Computes **PSNR, NIQE, FID, SSIM, LPIPS**. The IDS metric is available in `src/metrics/IDS.py` and can be enabled by uncommenting the IDS block in `evaluate.py` and pointing it at the ArcFace checkpoint.

```
src/metrics/
├── evaluate.py     Entry point.
├── PSNR.py
├── niqe.py
├── FID.py
├── IDS.py
└── ssim_lpips.py
```



