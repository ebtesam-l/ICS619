# Data

Layout expected by the training and evaluation scripts:

```
data/
├── celeba_hq/
│   ├── images/                     512x512 PNG/JPG.
│   └── splits/
│       ├── train.txt               One filename per line.
│       └── test.txt
└── saudi_dataset/                  Private. Follow the same layout.
    ├── images/
    └── splits/
        ├── train.txt
        └── test.txt
```

## CelebA HQ

Download from the official source and place 512x512 images under `celeba_hq/images/`. Use any community split or generate your own; the scripts only read the split files.

## Saudi Dataset

Private. The cleaning, alignment, and degradation code in `src/` runs on any folder of high-quality face crops following the layout above.

The split files (`train.txt` / `test.txt`) must contain no identity overlap.
