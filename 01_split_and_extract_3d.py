"""
Phase 1: Stratified Scaffold Split
-----------------------------------
Loads the pre-computed enriched 2D descriptor CSV, performs a
stratified Murcko-scaffold split (70 / 15 / 15), and writes
train_features.csv, val_features.csv, test_features.csv.

3D conformer generation has been intentionally removed: the enriched
CSV already contains rich 2D descriptor families (WHIM, RDF, MORSE,
GETAWAY proxies via RDKit 2D) and attempting per-molecule ETKDGv3
embedding on Windows deadlocks for the majority of multi-ring drug
scaffolds in this dataset.
"""

import os
import sys
import yaml
import json
import logging
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
import warnings

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def scaffold_split(df, seed=42):
    """
    Group molecules by Murcko scaffold, then stratify scaffold groups
    so that Label=0 (minority inactive) molecules appear in all three
    splits.  Returns (train_df, val_df, test_df).
    """
    scaffolds = {}
    for idx, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['canonical_smiles'])
            scf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else "Invalid"
        except Exception:
            scf = "Exception"
        scaffolds.setdefault(scf, []).append(idx)

    scaffold_keys = list(scaffolds.keys())

    # Scaffold-level label: 0 if the scaffold contains any inactive, else 1
    scaffold_labels = []
    for scf in scaffold_keys:
        has_inactive = (df.loc[scaffolds[scf], 'Label'] == 0).any()
        scaffold_labels.append(0 if has_inactive else 1)

    train_keys, temp_keys = train_test_split(
        scaffold_keys, test_size=0.30, random_state=seed, stratify=scaffold_labels
    )
    temp_labels = [scaffold_labels[scaffold_keys.index(k)] for k in temp_keys]
    val_keys, test_keys = train_test_split(
        temp_keys, test_size=0.50, random_state=seed, stratify=temp_labels
    )

    train_idx = [i for k in train_keys for i in scaffolds[k]]
    val_idx   = [i for k in val_keys   for i in scaffolds[k]]
    test_idx  = [i for k in test_keys  for i in scaffolds[k]]

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    logging.basicConfig(
        filename=config['pipeline']['log_file'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - Phase 1 - %(message)s'
    )

    seed = config['pipeline'].get('global_seed', 42)
    set_seed(seed)
    logging.info("Starting Scaffolds & 3D Conformer Logic")

    input_csv = config['pipeline']['input_csv']
    df = pd.read_csv(input_csv)
    logging.info(f"Loaded enriched CSV: {df.shape[0]} rows x {df.shape[1]} cols")

    # Stratified scaffold split
    train_df, val_df, test_df = scaffold_split(df, seed=seed)
    split_method = "Stratified Scaffold Split (RDKit)"
    logging.info(f"Splitting completed via {split_method}")
    logging.info(
        f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )

    # Label distribution sanity check
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_inactive = (part['Label'] == 0).sum()
        logging.info(f"  {name}: {len(part)} molecules, Label=0: {n_inactive}")
        print(f"  {name}: {len(part)} molecules | inactive (Label=0): {n_inactive}")

    # Write feature CSVs (all 2D descriptors already present in enriched CSV)
    train_df.to_csv("train_features.csv", index=False)
    val_df.to_csv("val_features.csv", index=False)
    test_df.to_csv("test_features.csv", index=False)
    logging.info("Feature CSVs written: train_features.csv, val_features.csv, test_features.csv")

    # Metadata
    meta = {
        "seed_value": seed,
        "split_mechanism": split_method,
        "splits": {
            "train": len(train_df),
            "val":   len(val_df),
            "test":  len(test_df),
        },
        "label_counts": {
            "train_inactive": int((train_df['Label'] == 0).sum()),
            "val_inactive":   int((val_df['Label']   == 0).sum()),
            "test_inactive":  int((test_df['Label']  == 0).sum()),
        }
    }
    with open(config['pipeline']['split_metadata'], 'w') as f:
        json.dump(meta, f, indent=4)

    logging.info("Phase 1 Successfully concluded.")
    print("Phase 1 complete. Feature CSVs ready.")


if __name__ == "__main__":
    main()
