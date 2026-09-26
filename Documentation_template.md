# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TEAM_NAME]  
**Team Members:** [TEAM_MEMBERS]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve Source-2/3 records to Source-1 reference entities with a three-stage pipeline: (1) script-agnostic normalization plus a transliteration dictionary learned from the training ground truth, (2) multi-pass sparse TF-IDF blocking in both directions (S1→pool and pool→S1), and (3) a two-stage LightGBM matcher whose second stage consumes relational statistics of first-stage probabilities, followed by a decision layer tuned directly on macro F0.5. The key structural insight is that Source 1 is deduplicated, so every Source-2/3 record has at most one parent; encoding that as features and as a hard "one parent" rule removes most false merges. Everything is CPU-only, uses no external data or services, and the only learned components are MIT-licensed LightGBM models.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on the training split (2.21M S1 entities, 5.03M S2 and 5.29M S3 records):

- Only 5.6% of S1 entities are singletons; the mean number of true matches is 3.45 (max 11). So recall matters despite the precision-weighted metric: an entity with three matches scores 0.71 if we return only one of them.
- Every S2/S3 id appears under exactly one S1 entity in the ground truth. Roughly 26% of pool records (1.34M per source) are unmatched distractors.
- About 9% of S2 names and 5% of S3 names are written in Indic scripts (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Odia); addresses frequently carry the state name in an Indic script. These are token-by-token transliterations of the Latin S1 name.
- Name noise: legal-suffix add/drop/move/bracket ("Private Manish Advisory ((Limited))"), word-order shuffles, multi-character typos ("Makreitrdng"), injected accents ("Ínvestments"), digit-for-letter ("0rthopedic"), duplicated tokens, filler tokens (Mr, The, Center, Services), bare domain names ("manishadvisory.com"), and DBA/FKA constructions ("Brixwex doing business as Wilk Delta Plus P.C.") including fully unrelated alias names that can only be matched through the address.
- Address noise: component reordering, abbreviation swaps (Street/St/Saint), state full-name vs code vs Indic script, filler tokens ("null", "N/A"), hash-noise in numbers ("##621", "S-##33"), house-number digit drops ("393" vs "3937"), missing house numbers, missing addresses (3.3%), city aliases (Bengaluru/Bangalore, Gurugram/Gurgaon), "Orissa"/"Odisha".
- Test composition differs from train: 47% India, 38% US, 15% France (unseen in training). The test pool has proportionally more S2/S3 records per S1 entity (5.8 vs 4.7).

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier + metric-tuned decision layer  
**Core Innovation:** (a) a transliteration dictionary learned purely from the ground truth alignment of Indic-script tokens to Latin S1 tokens; (b) a reverse (pool→S1) blocking pass and relational rank/gap features that exploit the one-parent structure; (c) a second-stage model over first-stage probabilities plus their within-group statistics, with the threshold and "one parent per record" rule tuned on macro F0.5 over held-out S1 entities.

---

## 3. Candidate Generation (Blocking)

All passes run within the same `country` label (treated as an open-set string; France gets its own block automatically). Text is normalized first: NFKC, learned transliteration, accent stripping, lower-casing, legal-suffix canonicalization, per-country address abbreviation and state-code dictionaries (US, India, France, generic fallback), filler removal, ordinal stripping, consecutive-duplicate removal.

- **Blocking keys used:**
  - Word TF-IDF (sublinear tf, tokens with document frequency > 2% pruned) over `name + address`, top-25 per S1 entity by cosine (sparse_dot_topn).
  - Word TF-IDF over `name` only, top-15 per S1 entity.
  - Character 3-gram TF-IDF over `name`, top-15 per S1 entity (catches typos).
  - Reverse pass: for every pool record, its top-3 S1 entities by `name + address` cosine (floor 0.15). Because each pool record has at most one parent, this guarantees the parent is a candidate even when the S1 side's list is crowded by near-duplicates.
- **Candidate pairs generated:** 91.7M for the test set (52.9 per S1 entity)
- **How you ensured true matches were not lost:** pair-level recall of the union measured on held-out training entities: 98.3% (per pass: full 96.6%, name 55.7%, char 29.5%, reverse 95.5%). K values were increased until recall plateaued.

---

## 4. Matching Model

**Features used:**
- Name features (on normalized full name, on the "core" name with legal suffixes removed, and on the DBA/FKA alias): Levenshtein ratio, token-sort and token-set ratios, partial ratio, Jaro-Winkler, space-less ratio (domain names), token Jaccard/coverage, shared-token count, first-token equality, character trigram Jaccard, token counts and lengths, legal-suffix agreement state, domain-name and alias flags.
- Address features: the same fuzzy suite on the normalized address, token Jaccard/coverage, alpha-token overlap (street/city words), house-number agreement state (equal / one missing / conflict / prefix), numeric-token Jaccard and shared count, postal-code agreement state, last-token (state/region) equality, emptiness flags.
- Name-rarity features (per country): how many S1 entities and how many pool records share the S1's core name, the same for the candidate's core name, and exact-core-name equality. A pool record whose name is unique among S1 entities can only belong to that entity, whereas a name shared by dozens of entities is usually a distractor; pairwise string similarity cannot see this.
- Blocking similarities from each pass and the number of passes that produced the pair.
- Relational features: rank, gap-to-best, margin-over-second and group size of a heuristic score and of the blocking cosine within the S1 entity's candidate list and within the pool record's candidate-parent list; source flag (S2/S3).
- Stage 2: stage-1 probability, its rank/gap/margin/count within the S1 group and within the pool-record group, sum of probabilities in the S1 group, counts above 0.5/0.8, plus a small subset of raw features.

**Model type:** LightGBM (binary, MIT licence), stage 1 on 76 pairwise features, stage 2 on stage-1 probability statistics. Stage-1 out-of-fold predictions (GroupKFold by S1 entity) are used to train stage 2. Training pairs come from 60,000 sampled S1 entities blocked against the full pool, plus "context rows": the candidates of every competing parent of those pool records, so the model sees the pool-side competition it will face at test time.  
**Threshold selection method:** grid search of the probability threshold, with and without the one-parent rule and a relative-gap rule, maximising macro F0.5 over held-out S1 entities (split by entity, blocked against the full pool).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9791 on 11,955 held-out training entities (stage 1 alone: 0.9760); per country US 0.9819, India 0.9746.
- **Common false positives (wrong merges):** pair precision is 0.995; the remaining wrong merges are (i) a different business at the same address with a heavily corrupted name ("Summit Inc." at the exact address of "Horizon Aura Center"), which is indistinguishable from an alias-renamed record, and (ii) same-name businesses whose house numbers differ by one digit ("220" vs "2200 Dorion Ave"), where the training labels themselves treat near-identical numbers inconsistently.
- **Common false negatives (missed matches):** pair recall is 0.94-0.95. Misses are dominated by pairs whose pool address is empty and whose name is shared by several S1 entities ("Cozy Yoga" vs "COZY YOGA CORPORATION" with no address): only the entity that actually owns the record is a match, and nothing in the pair reveals which one. The per-country name-rarity features (how many S1 entities / pool records share the core name) resolve the unique-name cases (99% matches, now kept) and leave the genuinely ambiguous ones, which the precision-weighted metric correctly prefers to drop. About 1.7% of true pairs are never generated as candidates, mostly Indic-script India names.

---

## 6. Conclusion

Two structural facts drive the solution: Source 1 is deduplicated (one parent per pool record), and business names repeat within a country (49% of S1 core names are shared). Exploiting the first through reverse blocking, relational rank/gap features and the one-parent decision rule, and the second through learned name-rarity features, raised the held-out macro F0.5 from 0.9714 (30k training entities, pairwise features only) to 0.9791 (60k entities plus context rows and rarity features) with pair precision 0.995. The pipeline is CPU-only, deterministic, uses no external data, and handles an unseen country (France) because every feature is script- and country-agnostic; the country label is only a blocking key.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/`: `prepare.py` (normalize + learn transliteration, cache parquet), `train.py` (blocking on a sample of S1 entities, features, two-stage LightGBM, decision tuning), `predict.py` (blocking → `output/candidate_pairs.tsv`, scoring + decision → `output/matching_results.tsv`, runs the validator). Supporting modules: `normalize.py`, `translit.py`, `blocking.py`, `features.py`, `evaluate.py`, `io_utils.py`, `config.py`. See the README for exact commands; the pipeline is CPU-only.

### B. Additional Results

Version history: v1 (30k entities) validation 0.9714 / public leaderboard 0.956; 60k entities + context rows 0.9753; v2 = 60k + context rows + name-rarity features + address-number normalization 0.9791 (this submission). Blocking: 51 candidates per training entity; recall of the union 98.3% (US 98.8%, India 97.0%). Test prediction produces 3.27 matches per S1 entity with 6.2% empty predictions, matching the training prior (3.45 matches, 5.6% singletons) once the 0.94 pair recall is accounted for. Compute: everything ran on a 16 GB laptop; training the final model takes ~40 min with cached reverse-blocking and full test scoring ~2.5 h (split by country across two machines with `predict.py --countries`).
