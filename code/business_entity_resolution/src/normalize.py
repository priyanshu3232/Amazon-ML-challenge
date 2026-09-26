"""Text normalization for business names and addresses.

Design goals
------------
* Country is an OPEN set: dictionaries are looked up by lower-cased country
  label; unknown labels get a safe generic dictionary. Nothing is hard-coded to
  {US, India}.
* Everything here is a deterministic rule or a mapping learned from the
  provided training data (see translit.py). No external services.
* Output is ASCII-only, lower-case, space-separated tokens so that all string
  similarity features are script-agnostic.
"""
import re
import unicodedata
from functools import lru_cache

from anyascii import anyascii

# --------------------------------------------------------------------------
# Legal suffixes (token -> canonical token). Applied to names of every country.
# --------------------------------------------------------------------------
LEGAL_SUFFIX = {
    "inc": "inc", "incorporated": "inc", "incorporation": "inc",
    "corp": "corp", "corporation": "corp", "corpn": "corp",
    "co": "co", "company": "co", "cie": "co", "compagnie": "co",
    "ltd": "ltd", "limited": "ltd", "ltee": "ltd",
    "pvt": "pvt", "private": "pvt",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pc": "pc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "sca": "sca", "gie": "gie",
    "ste": "ste", "societe": "ste", "sté": "ste",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "pty": "pty",
    "opc": "opc", "public": "public",
}
LEGAL_SET = set(LEGAL_SUFFIX.values())
# tokens that carry no identity information in names
NAME_FILLER = {"the", "mr", "mrs", "ms", "messrs", "m/s", "ms."}

ALIAS_RE = re.compile(
    r"\b(?:doing business as|d/b/a|dba|formerly known as|f/k/a|fka|"
    r"trading as|t/a|also known as|a/k/a|aka|now known as|n/k/a|nka)\b"
)
DOMAIN_RE = re.compile(
    r"^\s*(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)"
    r"\.(?:com|net|org|co|in|fr|io|biz|info|us|co\.in|co\.uk|eu)\s*$"
)
DOTTED_ABBR_RE = re.compile(r"\b((?:[a-z]\.){1,}[a-z]?)\.?(?=\s|$|[^a-z])")
ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
LEADING_ZERO_RE = re.compile(r"\b0+(?=\d)")   # zero-padded house/plot numbers
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
ZW_RE = re.compile("[​‌‍﻿]")

# --------------------------------------------------------------------------
# Address dictionaries per country (lower-case label -> {token: canonical}).
# Multi-word phrases are substituted on the string BEFORE tokenisation.
# --------------------------------------------------------------------------
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}
US_ADDR = {
    "street": "st", "str": "st", "avenue": "ave", "av": "ave", "road": "rd",
    "drive": "dr", "drv": "dr", "lane": "ln", "boulevard": "blvd", "blv": "blvd",
    "court": "ct", "place": "pl", "highway": "hwy", "parkway": "pkwy",
    "circle": "cir", "terrace": "ter", "trail": "trl", "square": "sq",
    "suite": "ste", "apartment": "apt", "building": "bldg", "floor": "fl",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "fort": "ft", "mount": "mt", "saint": "st", "route": "rte", "point": "pt",
    "heights": "hts", "junction": "jct", "expressway": "expy", "freeway": "fwy",
    "crossing": "xing", "center": "ctr", "centre": "ctr", "plaza": "plz",
    "creek": "crk", "ridge": "rdg", "valley": "vly", "village": "vlg",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as",
    "bihar": "br", "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga",
    "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od",
    "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg", "ts": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "uttaranchal": "uk", "ua": "uk", "west bengal": "wb", "delhi": "dl",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "ladakh": "la",
    "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn",
    "daman and diu": "dd", "lakshadweep": "ld",
    # common city spelling variants (either direction, just consistent)
    "bengaluru": "bangalore", "gurugram": "gurgaon", "bombay": "mumbai",
    "calcutta": "kolkata", "madras": "chennai", "poona": "pune",
    "trivandrum": "thiruvananthapuram", "cochin": "kochi", "mysuru": "mysore",
    "belagavi": "belgaum", "hubballi": "hubli", "mangaluru": "mangalore",
    "vizag": "visakhapatnam", "baroda": "vadodara", "prayagraj": "allahabad",
    "simla": "shimla", "kanpur nagar": "kanpur", "tiruchirappalli": "trichy",
    "tiruchirapalli": "trichy", "vijaywada": "vijayawada",
}
IN_ADDR = {
    "road": "rd", "street": "st", "opposite": "opp", "nr": "near",
    "floor": "fl", "flr": "fl", "ground": "gr", "sector": "sec", "phase": "ph",
    "block": "blk", "society": "soc", "apartment": "apt", "apartments": "apt",
    "building": "bldg", "industrial": "ind", "estate": "est",
    "extension": "extn", "ext": "extn", "taluk": "tq", "taluka": "tq",
    "district": "dist", "village": "vill", "post": "po", "tehsil": "teh",
    "east": "e", "west": "w", "north": "n", "south": "s", "lane": "ln",
    "colony": "col", "market": "mkt", "complex": "cplx", "tower": "twr",
    "towers": "twr", "chowk": "chowk", "nagar": "nagar", "marg": "marg",
    "layout": "layout", "cross": "cross", "main": "main",
}
FR_ADDR = {
    "rue": "rue", "r": "rue", "avenue": "av", "ave": "av", "boulevard": "bd",
    "bld": "bd", "blvd": "bd", "place": "pl", "chemin": "ch", "impasse": "imp",
    "route": "rte", "allee": "all", "cours": "crs", "quai": "quai",
    "faubourg": "fbg", "saint": "st", "sainte": "ste", "residence": "res",
    "batiment": "bat", "zone": "zone", "lieu": "lieu", "square": "sq",
    "passage": "pass", "sentier": "sent", "villa": "villa", "cite": "cite",
    "hameau": "ham", "esplanade": "espl", "promenade": "prom", "mont": "mt",
    "general": "gal", "marechal": "mal", "president": "pdt", "docteur": "dr",
    "professeur": "prof", "commandant": "cdt", "colonel": "col",
}
FR_REGIONS = {
    "ile de france": "idf", "nouvelle aquitaine": "naq",
    "auvergne rhone alpes": "ara", "hauts de france": "hdf",
    "provence alpes cote d azur": "paca", "grand est": "ges",
    "occitanie": "occ", "pays de la loire": "pdl", "bretagne": "bre",
    "normandie": "nor", "bourgogne franche comte": "bfc",
    "centre val de loire": "cvl", "corse": "cor",
}
GENERIC_ADDR = {
    "street": "st", "avenue": "ave", "road": "rd", "boulevard": "blvd",
    "floor": "fl", "building": "bldg", "apartment": "apt",
}
ADDR_DROP = {"no", "number", "num", "ndeg", "h", "hno", "null", "none", "nil",
             "de", "du", "des", "la", "le", "les", "l", "d", "of", "the",
             "at", "in", "and", "et"}
ADDR_PHRASES = {"h no": " ", "h.no": " ", "house no": " ", "door no": " ",
                "plot no": "plot ", "gala no": "gala ", "gut no": "gut ",
                "kh no": "kh ", "n/a": " ", "p o box": "po box",
                "p.o. box": "po box", "p.o box": "po box"}

COUNTRY_DICTS = {
    "us": (US_ADDR, US_STATES),
    "usa": (US_ADDR, US_STATES),
    "united states": (US_ADDR, US_STATES),
    "india": (IN_ADDR, IN_STATES),
    "in": (IN_ADDR, IN_STATES),
    "france": (FR_ADDR, FR_REGIONS),
    "fr": (FR_ADDR, FR_REGIONS),
}

_PHRASE_RES = {}
_ADDR_PHRASE_RE = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(re.escape(p) for p in sorted(ADDR_PHRASES, key=len, reverse=True)) + r")(?![a-z0-9])"
)


def _phrase_re(country_key):
    """Compiled regex for the multi-word phrases of a country (cached)."""
    if country_key not in _PHRASE_RES:
        _, phrases = COUNTRY_DICTS.get(country_key, (GENERIC_ADDR, {}))
        multi = sorted([p for p in phrases if " " in p], key=len, reverse=True)
        multi += sorted(ADDR_PHRASES, key=len, reverse=True)
        pat = "|".join(re.escape(p) for p in multi) if multi else r"(?!x)x"
        _PHRASE_RES[country_key] = re.compile(r"\b(?:" + pat + r")\b")
    return _PHRASE_RES[country_key]


# --------------------------------------------------------------------------
# Base string cleaning
# --------------------------------------------------------------------------
_translit_dict = {}


def set_translit_dict(d: dict) -> None:
    """Install a {non-latin token: latin token} mapping learned from training."""
    global _translit_dict
    _translit_dict = d
    _to_ascii.cache_clear()
    normalize_name.cache_clear()
    normalize_address.cache_clear()


@lru_cache(maxsize=2_000_000)
def _to_ascii(s: str) -> str:
    """NFKC + learned token transliteration + anyascii fallback + lower."""
    s = unicodedata.normalize("NFKC", s)
    s = ZW_RE.sub("", s)
    if s.isascii():
        return s.lower()
    if _translit_dict:
        parts = []
        for tok in s.split():
            key = tok.strip(",.;:()[]{}'\"-")
            rep = _translit_dict.get(key)
            parts.append(rep if rep is not None else tok)
        s = " ".join(parts)
    if not s.isascii():
        s = anyascii(s)
    return s.lower()


def _basic_clean(s: str) -> str:
    s = _to_ascii(s)
    s = s.replace("&", " and ").replace("+", " plus ").replace("@", " at ")
    s = s.replace("#", " ")
    s = DOTTED_ABBR_RE.sub(lambda m: m.group(1).replace(".", ""), s)
    return s


def _tokens(s: str):
    return [t for t in NON_ALNUM_RE.split(s) if t]


def _dedupe_consecutive(tokens):
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------
@lru_cache(maxsize=2_000_000)
def normalize_name(raw: str) -> tuple:
    """Return (full, core, alias, is_domain).

    full   : normalized name, legal suffixes canonicalized, fillers dropped
    core   : full minus legal-suffix tokens (identity part of the name)
    alias  : normalized alias after a DBA/FKA marker, or "" if none
    is_domain: 1 if the raw name is a bare domain like foo.com
    """
    s = _basic_clean(raw)
    is_domain = 0
    m = DOMAIN_RE.match(s)
    if m:
        is_domain = 1
        s = m.group(1).replace("-", " ")
    alias = ""
    parts = ALIAS_RE.split(s, maxsplit=1)
    if len(parts) == 2:
        left, right = parts[0].strip(), parts[1].strip()
        if left and right:
            s, alias = left, right
        else:
            s = left or right

    def _norm(x):
        toks = []
        for t in _tokens(x):
            if t in NAME_FILLER:
                continue
            toks.append(LEGAL_SUFFIX.get(t, t))
        toks = _dedupe_consecutive(toks)
        full = " ".join(toks)
        core = " ".join(t for t in toks if t not in LEGAL_SET) or full
        return full, core

    full, core = _norm(s)
    alias_full = _norm(alias)[1] if alias else ""
    return full, core, alias_full, is_domain


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------
@lru_cache(maxsize=2_000_000)
def normalize_address(raw: str, country: str) -> str:
    """Normalize an address for the given (open-set) country label."""
    if not raw or not raw.strip():
        return ""
    key = (country or "").strip().lower()
    abbr, phrases = COUNTRY_DICTS.get(key, (GENERIC_ADDR, {}))
    s = _basic_clean(raw)
    s = ORDINAL_RE.sub(r"\1", s)
    # punctuation-bearing fillers first (n/a, h.no, p.o. box ...)
    s = _ADDR_PHRASE_RE.sub(lambda m: " " + ADDR_PHRASES[m.group(0)] + " ", s)
    s = NON_ALNUM_RE.sub(" ", s)
    s = LEADING_ZERO_RE.sub("", s)
    pr = _phrase_re(key)
    merged = {**phrases, **ADDR_PHRASES}
    s = pr.sub(lambda m: " " + merged.get(m.group(0), m.group(0)) + " ", s)
    toks = []
    for t in _tokens(s):
        if t in ADDR_DROP:
            continue
        t = phrases.get(t, t)
        t = abbr.get(t, t)
        if t:
            toks.append(t)
    return " ".join(_dedupe_consecutive(toks))


def address_numbers(norm_addr: str) -> tuple:
    """Return (first numeric token, tuple of all tokens containing a digit)."""
    nums = tuple(t for t in norm_addr.split() if any(c.isdigit() for c in t))
    return (nums[0] if nums else ""), nums
