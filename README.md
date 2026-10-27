# Clinical-LQF

This repository provides model code and a DDR-based training and evaluation
example for five-class diabetic retinopathy grading. Requests for access to the
raw clinical data used in the study can be submitted to the corresponding author.

## Files

- `models/`: Clinical-LQF and ConvNeXt encoder.
- `losses/`: multi-task loss and target helpers in `models/heads.py`.
- `datasets/`: DDR CSV loader and joint image/map transforms.
- `utils/`: metrics and path expansion.
- `train.py`, `configs/ddr_clinical_lqf.yaml`: training and evaluation.
- `tools/`: prepare CSVs and export encoder weights.

## Install

Use Python 3.10 or newer. Install a matching PyTorch/torchvision pair for your
CUDA version from [PyTorch](https://pytorch.org/get-started/locally/), then run:

```bash
python -m pip install -r requirements.txt
```

## Prepare DDR

Obtain the grading images and official train/valid/test splits from
[DDR](https://github.com/nkicsl/DDR-dataset). Images follow the dataset's terms
and are not distributed here. Supply four soft lesion maps per image from a
frozen extractor: `<image_id>_MA.npy`, `_HE.npy`, `_EX.npy`, `_SE.npy`. Each map
must be a two-dimensional probability array in [0, 1], aligned to the RGB image.
The lesion extractor and private clinical data are not included.

From the repository root, replace the example paths and run:

```bash
python tools/build_ddr_master_csv.py --grading-root /path/to/DDR-dataset/DR_grading --pred-map-root /path/to/lesion_maps --out /path/to/ddr_prepared/ddr_grading_master.csv
python tools/generate_lesion_stats.py --master-csv /path/to/ddr_prepared/ddr_grading_master.csv --out /path/to/ddr_prepared/lesion_stats.csv
```

The scripts create output directories as needed. The master CSV contains `image_id`,
`image_path`, `split` (train/val/test), `grade_label` (0–4), and the four columns
`MA_pred_map_path`, `HE_pred_map_path`, `EX_pred_map_path`, `SE_pred_map_path`.
The helper writes absolute paths. The training loader also expands
`${DATA_ROOT}/...` and `${PACKAGE_ROOT}/...`; plain relative paths use the
current working directory. The stats generator uses the same path expansion.

The stats script writes `image_id` and the 18 features listed in
`datasets/ddr_dataset.py` (`STAT_COLUMNS`). Missing maps or stats raise an error.
For each lesion type,
`area_ratio` and `mean_prob` both use the mean probability over the map.

Set the data root to the directory containing both CSVs:

```bash
export DATA_ROOT=/path/to/ddr_prepared
```

In PowerShell, use `$env:DATA_ROOT = "D:\data\ddr_prepared"`.

The default configuration also requires local ConvNeXt encoder weights.
Export them once (requires network access or cached timm weights):

```bash
python tools/export_convnext_pretrained.py --model-name convnext_base.fb_in22k_ft_in1k_384 --out pretrained/convnext_base.fb_in22k_ft_in1k_384_encoder.pth
```

## Train and evaluate

Run from the repository root; adjust paths, batch size and workers in the YAML
for your machine. The default input size is 768 × 768; a CUDA GPU is recommended.

```bash
python train.py --config configs/ddr_clinical_lqf.yaml
python train.py --config configs/ddr_clinical_lqf.yaml --eval-only --checkpoint outputs/ddr_clinical_lqf/best_model.pth
```

The model uses MA evidence at N1, adds HE at N2, EX/SE at N3,
and deep image features with global statistics at N4. Thresholds are fitted
on validation data and then applied to the test split. Training uses inverse
class frequency sampling with unweighted classification loss by default.

Checkpoints, predictions and evaluation metrics are saved under
`outputs/ddr_clinical_lqf/`. No images, lesion maps or model weights are bundled.

## License

Code: [MIT](LICENSE). DDR images and third-party pretrained weights retain
their own terms.

DDR reference: Li, T. et al. Diagnostic assessment of deep learning algorithms
for diabetic retinopathy screening. *Information Sciences* **501**, 511–522
(2019). [DOI: 10.1016/j.ins.2019.06.011](https://doi.org/10.1016/j.ins.2019.06.011).
