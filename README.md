# MSNger

**MSNger** is a modular pipeline for preprocessing MRIs and constructing Morphometric Similarity Networks (MSNs). It supports flexible configurations including preprocessing options, feature selection, similarity measures, and the choice of atlas or template.

The pipeline integrates three Docker-based preprocessing tools into a single interface, culminating in MSN construction via the `MSNForge` module.

> **Note**: MSNger depends on external Docker images maintained separately. Monitor their updates when upgrading this pipeline.

For questions, contact `jor115@pitt.edu` or anyone at the Pediatric Imaging Research Center, Children's Hospital of Pittsburgh.

---

## Overview

MSNger builds ROI-by-ROI similarity matrices (MSNs) from multimodal MRI data:

| Modality | Docker Image | Output |
|---|---|---|
| T1w | `jor115/t1proc` | Radiomics features + volumes per ROI |
| rsfMRI | `jor115/sfp` | Functional connectivity matrix |
| DWI | `pennbbl/qsiprep:0.20.0` | Diffusion scalar maps (FA, MD, RD, AD) |
| MSN construction | `jor115/msnforge` | MSN similarity matrix CSV |

The entry point is `dockerCheat.py`, which orchestrates all Docker containers. `MSNForge/msnforge.py` can also be called directly if preprocessing is already complete.

---

## Requirements

- Python 3
- Docker
- Python packages: `python-on-whales`, `tqdm`
- A FreeSurfer license file (required for DWI preprocessing via QSIPrep)

---

## Usage

### Single subject

```bash
python3 dockerCheat.py \
  -p /path/to/bids/dataset \
  -sid sub-001 \
  -ses_id ses-01 \
  -o /path/to/derivatives \
  --fslicense /path/to/license.txt
```

### Batch — entire dataset

```bash
python3 dockerCheat.py \
  -p /path/to/bids/dataset \
  -o /path/to/derivatives \
  --fslicense /path/to/license.txt \
  --batch_whole_dataset
```

### Skip preprocessing, forge MSNs only

```bash
python3 dockerCheat.py \
  -p /path/to/bids/dataset \
  -sid sub-001 \
  -ses_id ses-01 \
  -o /path/to/derivatives \
  --fslicense /path/to/license.txt \
  --preprocess_only none
```

---

## Arguments

| Flag | Default | Description |
|---|---|---|
| `-p`, `--parentDir` | required | Path to BIDS dataset root |
| `-o`, `--outDir` | required | Output/derivatives directory |
| `--fslicense` | required | Path to FreeSurfer license |
| `-sid`, `--subject_id` | — | Subject ID (e.g. `sub-001`); required unless `--batch_whole_dataset` |
| `-ses_id`, `--session_id` | — | Session ID (e.g. `ses-01`) |
| `-tem`, `--template` | MNI152lin_T1_2mm | Template for registration |
| `-seg`, `--segment` | AALv2 | Atlas defining brain ROIs |
| `--preprocess_only` | `all` | Which modality to preprocess: `all`, `none`, `t1`, `dwi`, `bold` |
| `--features` | `all` | Feature set: `all`, `vol_radb_diff`, or `custom` |
| `--featureFile` | — | JSON file with custom features (required when `--features custom`) |
| `--similarity_measure` | `pearsonr` | Similarity metric: `pearsonr`, `cosine`, `inverse_euclidean` |
| `--batch_whole_dataset` | off | Process every subject/session in the dataset |
| `--skipforge` | off | Skip MSN construction; preprocessing only |
| `--supersizeme` | off | Compute MSNs for every feature set in the feature file |
| `--animalstyle` | off | Compute MSNs for every similarity measure (can combine with `--supersizeme`) |
| `--testmode` | off | Reduce iteration counts for faster debugging |

---

## Feature Sets

Feature sets are defined in JSON files. Two example files are provided:

- `features_fo-s-d-f.json` — first-order radiomics, shape radiomics, diffusion, and functional features
- `features_onlyStructural.json` — structural features only (volumes, radiomics, diffusion)

Each key in the JSON is a named feature set. Example:

```json
{
  "Rad-fo_diffusion": [
    "original_firstorder_Mean",
    "original_firstorder_Variance",
    "dti_fa",
    "md"
  ]
}
```

Use `--features custom --featureFile your_features.json` to specify a custom file.

---

## Output Structure

```
outDir/
└── MSNForge/
    └── sub-001/
        └── aal2/
            ├── consolidatedFeatures.csv
            ├── consolidatedFeatures_standardized.csv
            ├── MSN_feats-<featureset>_simfunc-<measure>.csv
            ├── <modality>_averages.csv
            └── ...
```

---

## Batch Example (from `runRad.py`)

```bash
python3 runRad.py
```

This script iterates over multiple sites and runs MSNger with structural-only features across all similarity measures and feature sets.

---

## License

See [LICENSE](LICENSE).
