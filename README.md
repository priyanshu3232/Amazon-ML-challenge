# Amazon ML Challenge 2026 — Business Entity Resolution

Matches Source-1 business records to their duplicates in Source-2/Source-3
(US, India, and unseen countries such as France). Scored by macro F0.5 per
Source-1 entity.

**Approach:** per-country name/address normalisation → learned Indic→Latin
transliteration → multi-pass TF-IDF top-K blocking → two-stage LightGBM
pair classifier → one-parent decision rule with a tuned threshold.
Validation macro F0.5 on held-out entities: 0.9714 (v1).

## Layout

| Path | Contents |
|---|---|
| `code/business_entity_resolution/src/` | Pipeline: `prepare.py` → `train.py` → `predict.py` |
| `code/business_entity_resolution/README.md` | How to run locally or on a remote machine |
| `code/business_entity_resolution/run_paramganga.sbatch` | SLURM job for a 48-core node |
| `Documentation_template.md` | Methodology write-up |
| `student_resource/utils/validate_submission.py` | Organisers' output validator |
| `make_submission_zip.sh` | Builds the submission zip |

The dataset, intermediate caches (`work/`), trained models and the output
TSVs are not in this repo; `prepare.py`/`train.py`/`predict.py` regenerate
them from the organisers' data placed in `student_resource/dataset/`.
