import pandas as pd

print("=== CHECKING NULL VALUES IN FIRST 100K ROWS ===")
for name, p in [("train_s1", "student_resource/dataset/train/train_source1.tsv"),
                ("train_s2", "student_resource/dataset/train/train_source2.tsv"),
                ("train_s3", "student_resource/dataset/train/train_source3.tsv"),
                ("test_s1", "student_resource/dataset/test/test_source1.tsv"),
                ("test_s2", "student_resource/dataset/test/test_source2.tsv"),
                ("test_s3", "student_resource/dataset/test/test_source3.tsv")]:
    df = pd.read_csv(p, sep="\t", nrows=100000)
    print(f"{name} shape: {df.shape} | nulls: {df.isnull().sum().to_dict()}")

print("\n=== SAMPLING REAL GROUND TRUTH MATCHES ===")
df_s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", nrows=50000).set_index("entity_id")
df_s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep="\t", nrows=100000).set_index("entity_id")
df_s3 = pd.read_csv("student_resource/dataset/train/train_source3.tsv", sep="\t", nrows=100000).set_index("entity_id")

gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", nrows=20000)
shown = 0
cross_country_matches = 0
total_checked = 0

for _, r in gt.iterrows():
    s1_id = r["source1_entity_id"]
    m = str(r["matched_entity_ids"])
    if pd.isna(m) or not m.strip():
        continue
    m_ids = [x.strip() for x in m.split(",") if x.strip()]
    if s1_id in df_s1.index:
        s1_row = df_s1.loc[s1_id]
        total_checked += 1
        matches_found = []
        for mid in m_ids:
            if mid in df_s2.index:
                row = df_s2.loc[mid]
                matches_found.append((mid, row["country"], row["business_name"], row["business_address"]))
                if row["country"] != s1_row["country"]:
                    cross_country_matches += 1
            elif mid in df_s3.index:
                row = df_s3.loc[mid]
                matches_found.append((mid, row["country"], row["business_name"], row["business_address"]))
                if row["country"] != s1_row["country"]:
                    cross_country_matches += 1
        
        if matches_found and shown < 5:
            shown += 1
            print(f"\n--- Example {shown}: S1 ID {s1_id} [{s1_row['country']}] ---")
            print(f"  S1 Name   : {s1_row['business_name']}")
            print(f"  S1 Address: {s1_row['business_address']}")
            for mid, c, name, addr in matches_found:
                print(f"  Match {mid} [{c}]:")
                print(f"     Name   : {name}")
                print(f"     Address: {addr}")

print(f"\nChecked {total_checked} matched S1 entities in sample. Cross-country matches found: {cross_country_matches}")
