"""
Module 1: Preprocessing & Cross-Lingual Normalization Engine.
Handles Unicode normalization, French diacritics, multi-country legal suffixes,
address standardization, and numeric token extraction.
"""

import re
import unicodedata
import pandas as pd
from typing import List, Tuple, Set

# Comprehensive Legal Entity Suffixes anchored to end of string (longest matches first)
ALL_LEGAL_SUFFIXES = [
    # Multi-word first
    r"limited\s+liability\s+company",
    r"private\s+limited",
    r"pvt\s+ltd",
    r"trading\s+co",
    # Single-word
    r"incorporated", r"corporation", r"enterprises", r"solutions", r"services", r"industries",
    r"limited", r"company", r"private",
    r"selarl", r"sarl", r"sasu", r"eurl", r"scop",
    r"corp", r"llc", r"ltd", r"llp", r"pvt", r"inc", r"sas", r"sci", r"snc", r"gie",
    r"co", r"lp", r"sa"
]
# Anchored to the end of the string to avoid stripping mid-string terms like 'co-op' or 'SA'
ANCHORED_LEGAL_REGEX = re.compile(r"\b(?:" + "|".join(ALL_LEGAL_SUFFIXES) + r")\s*$", re.IGNORECASE)

# English / General Address Abbreviations
ENGLISH_ADDR_REPLACEMENTS = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bdr\b": "drive",
    r"\bpkwy\b": "parkway",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bhwy\b": "highway",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bopp\b|\bopposite\s+to\b": "opposite",
    r"\bnr\b": "near",
}
COMPILED_EN_ADDR_REPLACEMENTS = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in ENGLISH_ADDR_REPLACEMENTS.items()]

# French-Specific Address Abbreviations (only applied to French records to avoid corrupting 'R Street', etc.)
FRENCH_ADDR_REPLACEMENTS = {
    r"\br\b": "rue",
    r"\bbd\b|\bbvd\b": "boulevard",
    r"\bav\b": "avenue",
    r"\ball\b": "allee",
    r"\bchem\b": "chemin",
    r"\bimp\b": "impasse",
    r"\bpl\b": "place",
}
COMPILED_FR_ADDR_REPLACEMENTS = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in FRENCH_ADDR_REPLACEMENTS.items()]


def strip_accents(text: str) -> str:
    """Normalize Unicode characters using NFKD and strip accents/diacritics."""
    if not isinstance(text, str):
        return ""
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def clean_business_name(raw_name: str) -> Tuple[str, str]:
    """
    Cleans raw business name:
    - Strips accents
    - Normalizes symbols (& -> and)
    - Strips legal entity designations anchored to the end of the string
    Returns (cleaned_name_without_suffix, extracted_suffix)
    """
    if not isinstance(raw_name, str):
        return "", ""

    text = strip_accents(raw_name).lower()
    text = text.replace("&", " and ").replace("@", " at ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Extract and strip legal suffix anchored to end of string only
    match = ANCHORED_LEGAL_REGEX.search(text)
    if match:
        suffix = match.group(0).strip()
        cleaned_name = text[:match.start()].strip()
    else:
        suffix = ""
        cleaned_name = text

    if not cleaned_name:
        cleaned_name = text  # fallback if stripping removed whole string

    return cleaned_name, suffix


def extract_numeric_tokens(address: str) -> List[str]:
    """
    Extracts building numbers, suite numbers, and PIN/ZIP codes.
    Preserves original extraction order from the text (not sorted lexicographically).
    """
    if not isinstance(address, str):
        return []
    # Match standalone digit sequences in order of appearance
    numbers = re.findall(r"\b\d{1,6}\b", address)
    seen = set()
    ordered_numbers = []
    for n in numbers:
        if n not in seen:
            seen.add(n)
            ordered_numbers.append(n)
    return ordered_numbers


def clean_address(raw_address: str, country: str = "") -> str:
    """
    Normalizes business address:
    - Strips accents
    - Standardizes street terms across English
    - Applies French street terms conditionally on country == 'FRANCE'
    - Removes punctuation and standardizes spacing
    """
    if not isinstance(raw_address, str):
        return ""

    text = strip_accents(raw_address).lower()
    text = text.replace("#", " unit ").replace("/", " ")

    # English / general replacements
    for pattern, replacement in COMPILED_EN_ADDR_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    # French-specific replacements only applied if country is FRANCE
    norm_country = str(country).strip().upper() if country else ""
    if norm_country == "FRANCE":
        for pattern, replacement in COMPILED_FR_ADDR_REPLACEMENTS:
            text = pattern.sub(replacement, text)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw TSV DataFrame into standardized cleaned schema.
    Columns in: entity_id, business_name, business_address, country
    Columns out: entity_id, country, clean_name, legal_suffix, clean_address,
                 extracted_numbers, clean_text
    """
    out_df = df.copy()

    # Clean Country
    out_df["country"] = out_df["country"].fillna("UNKNOWN").astype(str).str.strip().str.upper()

    b_names = out_df["business_name"].fillna("").astype(str).tolist()
    b_addrs = out_df["business_address"].fillna("").astype(str).tolist()
    b_countries = out_df["country"].tolist()

    # Process Names
    name_tuples = [clean_business_name(name) for name in b_names]
    out_df["clean_name"] = [t[0] for t in name_tuples]
    out_df["legal_suffix"] = [t[1] for t in name_tuples]

    # Process Addresses — pass country so French replacements are conditional
    out_df["clean_address"] = [
        clean_address(addr, country)
        for addr, country in zip(b_addrs, b_countries)
    ]
    out_df["extracted_numbers"] = [extract_numeric_tokens(addr) for addr in b_addrs]

    # Combined representation for lexical and dense embeddings
    out_df["clean_text"] = (
        out_df["clean_name"] + " | " + out_df["clean_address"] + " | " + out_df["country"]
    )

    return out_df


def load_and_preprocess_tsv(path: str, nrows: int = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Safely loads a tab-separated TSV file and applies full preprocessing.
    If use_cache is True and nrows is None, checks for a persisted pickle cache
    to avoid recomputing on subsequent pipeline runs.
    """
    import joblib
    from pathlib import Path

    path_obj = Path(path)
    suffix = f"_{nrows}" if nrows is not None else ""
    cache_path = path_obj.parent / f".{path_obj.stem}{suffix}_preprocessed.pkl"

    if use_cache and cache_path.exists():
        try:
            print(f"[Module 1 Cache] Loading cached preprocessed data from {cache_path.name}...")
            return joblib.load(cache_path)
        except Exception as e:
            print(f"[Module 1 Cache] Cache load failed ({e}), recomputing...")

    df = pd.read_csv(path, sep="\t", dtype=str, nrows=nrows)
    processed = preprocess_dataframe(df)

    if use_cache:
        try:
            joblib.dump(processed, cache_path, compress=3)
            print(f"[Module 1 Cache] Saved preprocessed cache to {cache_path.name}")
        except Exception:
            pass

    return processed


if __name__ == "__main__":
    # Smoke test on diverse multi-lingual test samples
    sample_data = {
        "entity_id": ["S1-001", "S2-002", "S3-003", "S1-004"],
        "business_name": [
            "Apple Inc.",
            "Boulangerie de l'Église SARL",
            "State Bank of India (SBI) Pvt Ltd",
            "Co-Op Market Co"
        ],
        "business_address": [
            "1 Infinite Loop, Ste 100, Cupertino, CA 95014",
            "14 Rue de la Paix, 75002 Paris",
            "Opp. SBI ATM, M.G. Rd, Bengaluru 560001",
            "123 R Street, Building 9, Suite 80"
        ],
        "country": ["US", "France", "India", "US"]
    }
    df = pd.DataFrame(sample_data)
    processed = preprocess_dataframe(df)
    print("Module 1 Smoke Test PASSED!")
    print(processed[["entity_id", "country", "clean_name", "legal_suffix", "clean_address", "extracted_numbers"]])
