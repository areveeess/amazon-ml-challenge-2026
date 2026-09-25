"""
Module 1: Preprocessing & Cross-Lingual Normalization Engine.
Handles Unicode normalization, French diacritics, multi-country legal suffixes,
address standardization, and numeric token extraction.
"""

import re
import unicodedata
import pandas as pd
from typing import List, Tuple, Set

# Comprehensive Legal Entity Suffixes by Region
LEGAL_PATTERNS = [
    # US / International
    r"\b(incorporated|inc|corporation|corp|limited\s+liability\s+company|llc|co|company|ltd|limited|lp|llp)\b",
    # India
    r"\b(pvt\s+ltd|private\s+limited|pvt|private|enterprises|solutions|services|industries|trading\s+co)\b",
    # France (Test Set Domain Shift)
    r"\b(sarl|sas|sasu|sa|eurl|sci|snc|gie|selarl|scop)\b"
]
LEGAL_REGEX = re.compile("|".join(LEGAL_PATTERNS), re.IGNORECASE)

# Address Abbreviations
ADDRESS_REPLACEMENTS = {
    # English / US / India
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
    # French
    r"\br\b": "rue",
    r"\bbd\b|\bbvd\b": "boulevard",
    r"\bav\b": "avenue",
    r"\ball\b": "allee",
    r"\bchem\b": "chemin",
    r"\bimp\b": "impasse",
    r"\bpl\b": "place"
}
COMPILED_ADDR_REPLACEMENTS = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in ADDRESS_REPLACEMENTS.items()]

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
    - Strips legal entity designations
    Returns (cleaned_name_without_suffix, extracted_suffix)
    """
    if not isinstance(raw_name, str):
        return "", ""
    
    text = strip_accents(raw_name).lower()
    text = text.replace("&", " and ").replace("@", " at ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Extract legal suffix
    matches = LEGAL_REGEX.findall(text)
    suffix = ""
    if matches:
        last = matches[-1]
        suffix = last if isinstance(last, str) else [g for g in last if g][-1]
    
    # Strip suffix from core name
    cleaned_name = LEGAL_REGEX.sub(" ", text)
    cleaned_name = re.sub(r"\s+", " ", cleaned_name).strip()
    if not cleaned_name:
        cleaned_name = text  # fallback if stripping removed whole string
        
    return cleaned_name, suffix

def extract_numeric_tokens(address: str) -> List[str]:
    """Extracts building numbers, suite numbers, and 5-6 digit PIN/ZIP codes."""
    if not isinstance(address, str):
        return []
    # Match standalone digit sequences
    numbers = re.findall(r"\b\d{1,6}\b", address)
    return sorted(list(set(numbers)))

def clean_address(raw_address: str) -> str:
    """
    Normalizes business address:
    - Strips accents
    - Standardizes street terms across English and French
    - Removes punctuation and standardizes spacing
    """
    if not isinstance(raw_address, str):
        return ""
    
    text = strip_accents(raw_address).lower()
    text = text.replace("#", " unit ").replace("/", " ")
    
    for pattern, replacement in COMPILED_ADDR_REPLACEMENTS:
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
    
    # Process Names
    name_tuples = [clean_business_name(str(name)) for name in out_df["business_name"]]
    out_df["clean_name"] = [t[0] for t in name_tuples]
    out_df["legal_suffix"] = [t[1] for t in name_tuples]
    
    # Process Addresses
    out_df["clean_address"] = [clean_address(str(addr)) for addr in out_df["business_address"]]
    out_df["extracted_numbers"] = [extract_numeric_tokens(str(addr)) for addr in out_df["business_address"]]
    
    # Combined representation for lexical and dense embeddings
    out_df["clean_text"] = (
        out_df["clean_name"] + " | " + out_df["clean_address"] + " | " + out_df["country"]
    )
    
    return out_df

def load_and_preprocess_tsv(path: str) -> pd.DataFrame:
    """Safely loads a tab-separated TSV file and applies full preprocessing."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    return preprocess_dataframe(df)

if __name__ == "__main__":
    # Smoke test on diverse multi-lingual test samples
    sample_data = {
        "entity_id": ["S1-001", "S2-002", "S3-003"],
        "business_name": ["Apple Inc.", "Boulangerie de l'Église SARL", "State Bank of India (SBI) Pvt Ltd"],
        "business_address": ["1 Infinite Loop, Ste 100, Cupertino, CA 95014", "14 Rue de la Paix, 75002 Paris", "Opp. SBI ATM, M.G. Rd, Bengaluru 560001"],
        "country": ["US", "France", "India"]
    }
    df = pd.DataFrame(sample_data)
    processed = preprocess_dataframe(df)
    print("Module 1 Smoke Test PASSED!")
    print(processed[["entity_id", "country", "clean_name", "legal_suffix", "extracted_numbers"]])
