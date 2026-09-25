"""Learn a token-level transliteration dictionary from the TRAINING ground truth.

About 9% of Source-2/3 names (and the state part of many addresses) are written
in Indic scripts (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Odia).
Generic transliteration (anyascii) drops inherent vowels ("lksmi" for Laxmi),
which weakens string similarity. Because the ground truth pairs each such
record with its Latin-script Source-1 entity, we can align tokens and learn
{indic token -> latin token}. This uses only the provided training data.
"""
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from normalize import ZW_RE, normalize_address

_LATIN_TOK = re.compile(r"[a-z0-9]+")
_STRIP = ",.;:()[]{}'\"-"


def _raw_tokens(s: str):
    s = ZW_RE.sub("", unicodedata.normalize("NFKC", s))
    return [t.strip(_STRIP) for t in s.split() if t.strip(_STRIP)]


def _is_nonlatin(tok: str) -> bool:
    return not tok.isascii()


def learn_translit(s1_df, pool_df, gt: dict, min_count=2, min_ratio=0.3) -> dict:
    """Return {non-latin token: latin token} learned from matched pairs."""
    parent = {}
    for s1, ids in gt.items():
        for m in ids:
            parent[m] = s1
    s1_name = dict(zip(s1_df.entity_id, s1_df.business_name))
    s1_addr = dict(zip(s1_df.entity_id, s1_df.business_address))
    s1_ctry = dict(zip(s1_df.entity_id, s1_df.country))

    counts = defaultdict(Counter)
    occ = Counter()                      # records containing each non-latin token
    it = zip(pool_df.entity_id, pool_df.business_name,
             pool_df.business_address, pool_df.country)
    for eid, name, addr, ctry in it:
        if name.isascii() and addr.isascii():
            continue
        s1 = parent.get(eid)
        if s1 is None:
            continue
        if not name.isascii():
            ptoks = _raw_tokens(name)
            ltoks = _LATIN_TOK.findall(s1_name[s1].lower())
            for p in set(ptoks):
                if _is_nonlatin(p):
                    occ[p] += 1
            if len(ptoks) == len(ltoks):
                for p, l in zip(ptoks, ltoks):
                    if _is_nonlatin(p):
                        counts[p][l] += 1
            else:
                for p in ptoks:
                    if _is_nonlatin(p):
                        for l in set(ltoks):
                            counts[p][l] += 1
        if not addr.isascii():
            ptoks = _raw_tokens(addr)
            ltoks = normalize_address(s1_addr[s1], s1_ctry[s1]).split()
            nonlat = [p for p in ptoks if _is_nonlatin(p)]
            for p in set(nonlat):
                occ[p] += 1
            if len(ptoks) == len(ltoks):
                for p, l in zip(ptoks, ltoks):
                    if _is_nonlatin(p):
                        counts[p][l] += 1
            else:
                for p in nonlat:
                    for l in set(ltoks):
                        counts[p][l] += 1
    out = {}
    for p, c in counts.items():
        (best, n), = c.most_common(1)
        # ratio = fraction of records containing p whose counterpart has `best`
        if n >= min_count and n / max(1, occ[p]) >= min_ratio:
            out[p] = best
    return out


def save_translit(d: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)


def load_translit(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)
