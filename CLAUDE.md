# MSNger — Claude Code Context

## What this project does

MSNger builds **Morphometric Similarity Networks (MSNs)**: ROI-by-ROI similarity matrices derived from multimodal MRI (T1w, rsfMRI, DWI). Each node is a brain region (from an atlas like AALv2); edges encode similarity across a chosen feature vector.

The pipeline has two phases:
1. **Preprocessing** — three Docker containers handle T1w, rsfMRI, and DWI independently
2. **Forging** — `MSNForge/msnforge.py` collects preprocessed outputs, consolidates features, z-scores them, and computes the similarity matrix

## Entry points

- `dockerCheat.py` — main CLI; orchestrates Docker containers for preprocessing and MSNForge
- `MSNForge/msnforge.py` — MSN construction only; called inside the `jor115/msnforge` container

## Docker images used

| Image | Purpose |
|---|---|
| `jor115/t1proc` | T1w preprocessing + radiomics feature extraction |
| `jor115/sfp` | rsfMRI preprocessing + functional connectivity matrix |
| `pennbbl/qsiprep:0.20.0` | DWI preprocessing + diffusion scalar maps (FA, MD, RD, AD) |
| `jor115/msnforge` | MSN construction (wraps `msnforge.py`) |

## Key files

| File | Role |
|---|---|
| `dockerCheat.py` | Top-level pipeline orchestrator |
| `MSNForge/msnforge.py` | Feature consolidation + MSN computation |
| `MSNForge/registration.py` | FSL FLIRT + ANTs SyN atlas-to-diffusion-map registration |
| `MSNForge/utils.py` | Filename helpers (`split_full_extension`, `appendToBasename`, `extract_modality_suffix`) |
| `features_fo-s-d-f.json` | Named feature sets: first-order radiomics, shape, diffusion, functional |
| `features_onlyStructural.json` | Named feature sets: volumes, radiomics, diffusion (no functional) |
| `qgi_scalar_export.json` | QSIRecon reconstruction spec (scalar export config) |

## Data assumptions

- Input data should be **BIDS-compatible** (subject dirs named `sub-*`, sessions named `ses-*`)
- Preprocessed T1 outputs land in `outDir/RadT1cal_Features/<subject_id>/`
- Preprocessed BOLD outputs land in `outDir/Sim_Funky_Pipeline/<subject_id>/`
- Preprocessed DWI outputs land in `outDir/qsirecon/<subject_id>/`
- `MSNForge` copies these into `outDir/MSNForge/<subject_id>/<atlas_name>/` before processing

## Similarity measures

Three options, all symmetric pairwise over ROI feature vectors:
- `pearsonr` (default) — Pearson correlation
- `cosine` — cosine similarity
- `inverse_euclidean` — `1 / (1 + ||a - b||)`

## Feature sets

Defined as named lists of column names in a JSON file. The column names must match what appears in the consolidated CSV after preprocessing. Use `--supersizeme` to run all feature sets in a file; use `--animalstyle` to run all similarity measures.

`features_fo-s-d-f.json` contains all 7 combinations of first-order radiomics (fo), shape radiomics (s), and diffusion (d) as standalone sets, plus `Functional` alone and the full `fo+s+d+f` group. Functional features are intentionally only included in the largest combined set.

`features_onlyStructural.json` is for datasets without fMRI. It covers volume, first-order radiomics, and diffusion combinations, plus `Radiomic_fo_basic` (4 features), `Radiomic_fo_all` (18 first-order features), `Radiomic_shape` (4 shape features), and `Diffusion` as standalone sets.

## Registration pipeline (in `registration.py`)

1. FSL FLIRT: affine-align template → diffusion FA map
2. Apply same affine to atlas
3. ANTs SyN: nonlinear-align FLIRT-warped template → FA map
4. Apply ANTs transform to affine-warped atlas (nearest-neighbor interpolation)
5. Clean up intermediates

## Output

Final MSN saved as CSV: `MSN_feats-<featureset>_simfunc-<measure>.csv`  
Zero diagonal; symmetric; values in [-1, 1] for Pearson, [0, 1] for cosine/inverse-euclidean.
