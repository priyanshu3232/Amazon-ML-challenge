# Business Entity Resolution — pipeline

End-to-end, CPU-only pipeline that regenerates `output/candidate_pairs.tsv` and
`output/matching_results.tsv` from the challenge data.

## Layout

```
code/business_entity_resolution/
  src/
    config.py      paths and tunable parameters (blocking K, LightGBM params)
    io_utils.py    TSV reading/writing (tab separated, no quoting, no NA parsing)
    normalize.py   per-country name/address normalization (open-set country labels)
    translit.py    learns Indic-script -> Latin token dictionary from train ground truth
    blocking.py    multi-pass sparse TF-IDF top-K candidate generation (both directions)
    features.py    pairwise string/address/relational features, stage-2 features
    evaluate.py    macro F0.5 metric, decision layer (threshold + one-parent rule), tuning
    prepare.py     step 1: normalize a split and cache parquet
    train.py       step 2: block a sample of train S1 entities, train 2-stage LightGBM,
                   tune the decision layer on held-out S1 entities
    predict.py     step 3: block + score the test split, write both output files, validate
  requirements.txt
```

Expected repository layout (paths are resolved relative to this folder; override
with `ER_ROOT=/path/to/repo`):

```
<repo>/student_resource/dataset/{train,test}/...   challenge data (as distributed)
<repo>/work/                                       intermediate parquet, models, logs
<repo>/output/                                     submission files
```

## Environment

Python 3.12. Install pinned dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

macOS only: LightGBM needs the OpenMP runtime (`brew install libomp`).

## Reproduce end to end

```bash
cd code/business_entity_resolution/src
python prepare.py --split train          # ~3 min: learn transliteration dict, normalize, cache
python prepare.py --split test           # ~3 min
python train.py --n-queries 400000       # block + features + 2-stage LightGBM + decision tuning
python predict.py                        # blocking -> candidate_pairs.tsv, matching -> matching_results.tsv
```

`predict.py` finishes by running `student_resource/utils/validate_submission.py`
on the two output files.

Quick smoke test: `python train.py --n-queries 60000 --tag _dev` then
`python predict.py --limit 50000 --tag _dev` (writes to `work/smoke/`).

## Hardware used

Development on an Apple M5 laptop (16 GB). The full pipeline is CPU-only; a
machine with 32+ GB RAM and 16+ cores runs the full test prediction in
about an hour. No GPU is required.

## Fair-play statement

No external data, APIs, or services are used. All dictionaries in
`normalize.py` are static rules (abbreviations, state codes); the
transliteration dictionary is learned only from `train_ground_truth.tsv`.
