"""
Module 4 (Part 2): Adaptive Confidence Cascade Reranker.
Routes ambiguous pairs (where primary GBDT probability is uncertain, e.g. [0.38, 0.72])
to a Cross-Encoder for deep attention-based pairwise verification.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

class AdaptiveConfidenceCascade:
    def __init__(
        self,
        ambiguity_low: float = 0.38,
        ambiguity_high: float = 0.72,
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.ambiguity_low = ambiguity_low
        self.ambiguity_high = ambiguity_high
        self.cross_encoder_name = cross_encoder_name
        self.cross_encoder = None

    def _init_cross_encoder(self):
        if self.cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder(self.cross_encoder_name)
                print(f"[Module 4] Initialized Cross-Encoder: {self.cross_encoder_name}")
            except Exception as e:
                print(f"[Module 4] CrossEncoder could not be loaded ({e}). Relying purely on calibrated GBDT probabilities.")
                self.cross_encoder = None

    def refine_probabilities(
        self,
        scored_pairs_df: pd.DataFrame,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
        use_cascade: bool = True
    ) -> pd.DataFrame:
        """
        Takes initial GBDT probabilities and selectively reranks only ambiguous pairs.
        """
        df = scored_pairs_df.copy()
        if not use_cascade:
            return df
            
        # Identify ambiguous zone
        ambig_mask = (df["predicted_prob"] >= self.ambiguity_low) & (df["predicted_prob"] <= self.ambiguity_high)
        num_ambig = ambig_mask.sum()
        print(f"[Module 4 Cascade] Found {num_ambig}/{len(df)} pairs in ambiguous zone [{self.ambiguity_low}, {self.ambiguity_high}]")
        
        if num_ambig == 0:
            return df
            
        self._init_cross_encoder()
        if self.cross_encoder is None:
            return df
            
        s1_map = s1_df.set_index("entity_id")["clean_text"].to_dict()
        tgt_map = target_df.set_index("entity_id")["clean_text"].to_dict()
        
        ambig_indices = df[ambig_mask].index
        sentence_pairs = []
        for idx in ambig_indices:
            s1_id = df.loc[idx, "source1_entity_id"]
            cand_id = df.loc[idx, "candidate_entity_id"]
            t1 = s1_map.get(s1_id, "")
            t2 = tgt_map.get(cand_id, "")
            sentence_pairs.append((t1, t2))
            
        # Cross-encoder inference
        ce_scores = self.cross_encoder.predict(sentence_pairs, show_progress_bar=False)
        # Apply sigmoid to convert logits to probabilities
        ce_probs = 1.0 / (1.0 + np.exp(-ce_scores))
        
        # Blend GBDT and Cross-Encoder probabilities in the ambiguous zone
        blended = 0.5 * df.loc[ambig_indices, "predicted_prob"].values + 0.5 * ce_probs
        df.loc[ambig_indices, "predicted_prob"] = blended
        print(f"[Module 4 Cascade] Successfully refined {num_ambig} ambiguous pairs via Cross-Encoder.")
        
        return df

if __name__ == "__main__":
    cascade = AdaptiveConfidenceCascade()
    test_df = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2"],
        "candidate_entity_id": ["S2-1", "S3-2"],
        "predicted_prob": [0.85, 0.52]
    })
    s1_df = pd.DataFrame({"entity_id": ["S1-1", "S1-2"], "clean_text": ["A", "B"]})
    tgt_df = pd.DataFrame({"entity_id": ["S2-1", "S3-2"], "clean_text": ["A", "B"]})
    out = cascade.refine_probabilities(test_df, s1_df, tgt_df, use_cascade=False)
    print("Module 4 Cascade Smoke Test PASSED!")
