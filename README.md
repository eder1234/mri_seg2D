# MRI Segmentation Utilities

This repository contains scripts to train and run UNet models for cerebrospinal fluid (CSF) segmentation from phase--contrast MRI volumes. Models are implemented with [PyTorch](https://pytorch.org/) and [MONAI](https://monai.io/).

## Repository structure

- `build_data.py` – convert the raw export into the `data/` folder used by the training scripts.
- `check_shapes.py` – sanity‑check that each `mag.npy`, `phase.npy` and `mask.npy` file has the expected dimensions.
- `configs/` – YAML configuration files describing dataset location and model hyper‑parameters.
- `datasets/` – dataset loader used by the training code.
- `models/` – 2‑D and 3‑D UNet implementations.
- `train.py` – train a model for a single cross‑validation fold.
- `infer.py` / `infer_2d.py` – run inference with a trained checkpoint.
- `ensemble.py` – average the probability maps of several checkpoints.
- `viz_overlay*.py`, `viz_random_sample.py` – helper scripts to visualise predictions.

## Installation

Install the Python dependencies with

```bash
pip install -r requirements.txt
```

## Data format

After running `build_data.py` the `data/` directory should have one folder per patient, e.g.

```
data/
└─ Patient_X/
   ├─ mag.npy    # (D, H, W)
   ├─ phase.npy
   ├─ mask.npy   # ground truth mask
   └─ meta.json  # optional metadata
```

`check_shapes.py` can be used to print the shapes of all volumes and ensure the dataset was created correctly.

## Training

Train a model for a specific fold by providing a configuration file:

```bash
python train.py --config configs/unet2d_multiframe.yaml --fold 0
```

Checkpoints are written under `runs/`.

## Inference

Run inference on a single patient folder:

```bash
python infer.py --ckpt <checkpoint.pt> --config configs/unet3d_baseline.yaml \
                --input data/Patient_X --out Patient_X.nii.gz
```

For the 2‑D models use `infer_2d.py` instead. Predicted masks are saved as NIfTI files.

Multiple checkpoints can be averaged with

```bash
python ensemble.py --config configs/ensemble.yaml --input data/Patient_X --out Patient_X.npy
```

## Visualization

`viz_overlay.py` and `viz_overlay_dual.py` overlay a predicted mask on the magnitude and/or phase images. `viz_random_sample.py` shows a random validation subject for a trained model.

## License

All code in this repository is released under the MIT license unless noted otherwise.
