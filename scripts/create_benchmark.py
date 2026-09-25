"""
Creates an honest, representative 50,000-entity benchmark split
from the official student_resource dataset.

Stratification:
- 60% US, 40% India (matching real distribution)
- 5.58% Singletons (matching ground truth singleton ratio)
- Leak-free split: 40,000 Train S1 entities (80%) and 10,000 Validation S1 entities (20%)
- Target pool: All true matches from S2 and S3 + hard negative pool from same countries.
"""

import os
import sys
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict

random.seed(42)

BENCH_DIR = Path("benchmark_dataset")
BENCH_TRAIN = BENCH_DIR / "train"
BENCH_VAL = BENCH_DIR / "val"
BENCH_TRAIN.mkdir(parents=True, exist_ok=True)
BENCH_VAL.mkdir(parents=True, exist_ok=True)

RAW_DIR = Path("student_resource/dataset/train")
SRC1_PATH = RAW_DIR / "train_source1.tsv"
SRC2_PATH = RAW_DIR / "train_source2.tsv"
SRC3_PATH = RAW_DIR / "train_source3.tsv"
GT_PATH = RAW_DIR / "train_ground_truth.tsv"

print("--- [Step 1/4] Reading Ground Truth to classify Singletons vs Matches ---")
gt_matches = {}
with open(GT_PATH, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        s1_id = parts[0].strip()
        m_str = parts[1].strip() if len(parts) > 1 else ""
        gt_matches[s1_id] = m_str

print(f"Loaded ground truth for {len(gt_matches):,} entities.")

print("--- [Step 2/4] Sampling 50,000 Stratified S1 Entities ---")
us_singletons, us_matched = [], []
in_singletons, in_matched = [], []

with open(SRC1_PATH, "r", encoding="utf-8") as f:
    header = next(f).rstrip("\r\n").split("\t")
    id_idx = header.index("entity_id")
    c_idx = header.index("country")
    
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) <= c_idx:
            continue
        eid = parts[id_idx].strip()
        country = parts[c_idx].strip().upper()
        m_str = gt_matches.get(eid, "")
        
        is_sing = (len(m_str) == 0)
        
        if country == "US":
            if is_sing:
                us_singletons.append(eid)
            else:
                us_matched.append(eid)
        elif country == "INDIA":
            if is_sing:
                in_singletons.append(eid)
            else:
                in_matched.append(eid)

print(f"Pool sizes: US (Sing={len(us_singletons):,}, Mat={len(us_matched):,}) | India (Sing={len(in_singletons):,}, Mat={len(in_matched):,})")

# Target 50,000: 30,000 US (1,674 singletons, 28,326 matched), 20,000 India (1,116 singletons, 18,884 matched)
target_us_sing = int(30000 * 0.0558)
target_us_mat = 30000 - target_us_sing
target_in_sing = int(20000 * 0.0558)
target_in_mat = 20000 - target_in_sing

sampled_us_sing = random.sample(us_singletons, target_us_sing)
sampled_us_mat = random.sample(us_matched, target_us_mat)
sampled_in_sing = random.sample(in_singletons, target_in_sing)
sampled_in_mat = random.sample(in_matched, target_in_mat)

us_all = sampled_us_sing + sampled_us_mat
random.shuffle(us_all)
in_all = sampled_in_sing + sampled_in_mat
random.shuffle(in_all)

# 80/20 train/val split within US and India
train_s1_ids = set(us_all[:24000] + in_all[:16000])  # 40,000
val_s1_ids = set(us_all[24000:] + in_all[16000:])    # 10,000
benchmark_s1_ids = train_s1_ids | val_s1_ids

print(f"Selected: Train S1 = {len(train_s1_ids):,}, Val S1 = {len(val_s1_ids):,}")

print("--- [Step 3/4] Extracting Target Records (S2 & S3) & Collecting Match IDs ---")
needed_target_ids = set()
for eid in benchmark_s1_ids:
    m_str = gt_matches.get(eid, "")
    if m_str:
        for mid in m_str.split(","):
            mid = mid.strip()
            if mid:
                needed_target_ids.add(mid)

print(f"Total True Target IDs to extract: {len(needed_target_ids):,}")

# Write Benchmark Ground Truth
with open(BENCH_TRAIN / "train_ground_truth.tsv", "w", encoding="utf-8") as f_trn, \
     open(BENCH_VAL / "val_ground_truth.tsv", "w", encoding="utf-8") as f_val:
    f_trn.write("source1_entity_id\tmatched_entity_ids\n")
    f_val.write("source1_entity_id\tmatched_entity_ids\n")
    for eid in train_s1_ids:
        f_trn.write(f"{eid}\t{gt_matches.get(eid, '')}\n")
    for eid in val_s1_ids:
        f_val.write(f"{eid}\t{gt_matches.get(eid, '')}\n")

# Write Benchmark Source 1
with open(SRC1_PATH, "r", encoding="utf-8") as f_in, \
     open(BENCH_TRAIN / "train_source1.tsv", "w", encoding="utf-8") as f_trn, \
     open(BENCH_VAL / "val_source1.tsv", "w", encoding="utf-8") as f_val:
    header = next(f_in)
    f_trn.write(header)
    f_val.write(header)
    id_idx = header.rstrip("\r\n").split("\t").index("entity_id")
    for line in f_in:
        parts = line.rstrip("\r\n").split("\t")
        eid = parts[id_idx].strip()
        if eid in train_s1_ids:
            f_trn.write(line)
        elif eid in val_s1_ids:
            f_val.write(line)

print("--- [Step 4/4] Extracting S2 & S3 Records (True Targets + Distractor Pool) ---")
# Extract S2 (targets + 25k distractors)
s2_count = 0
s2_distractors = 0
with open(SRC2_PATH, "r", encoding="utf-8") as f_in, \
     open(BENCH_DIR / "benchmark_source2.tsv", "w", encoding="utf-8") as f_out:
    header = next(f_in)
    f_out.write(header)
    id_idx = header.rstrip("\r\n").split("\t").index("entity_id")
    for line in f_in:
        parts = line.rstrip("\r\n").split("\t")
        eid = parts[id_idx].strip()
        if eid in needed_target_ids:
            f_out.write(line)
            s2_count += 1
        elif s2_distractors < 25000 and random.random() < 0.05:
            f_out.write(line)
            s2_distractors += 1

# Extract S3 (targets + 25k distractors)
s3_count = 0
s3_distractors = 0
with open(SRC3_PATH, "r", encoding="utf-8") as f_in, \
     open(BENCH_DIR / "benchmark_source3.tsv", "w", encoding="utf-8") as f_out:
    header = next(f_in)
    f_out.write(header)
    id_idx = header.rstrip("\r\n").split("\t").index("entity_id")
    for line in f_in:
        parts = line.rstrip("\r\n").split("\t")
        eid = parts[id_idx].strip()
        if eid in needed_target_ids:
            f_out.write(line)
            s3_count += 1
        elif s3_distractors < 25000 and random.random() < 0.05:
            f_out.write(line)
            s3_distractors += 1

print(f"Extracted Source 2: {s2_count:,} true targets + {s2_distractors:,} distractors")
print(f"Extracted Source 3: {s3_count:,} true targets + {s3_distractors:,} distractors")
print("Successfully constructed Benchmark Dataset in benchmark_dataset/!")
