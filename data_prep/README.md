# Data Preparation & Provenance

This directory is intended to house the initial data collection and preparation scripts (Phases 1-4 of the broader workflow) that precede the machine learning pipeline in the root of this repository.

## Upstream Pipeline Context

The provided input file `phase5_rdkit_2D_descriptors_enriched.csv` was generated through an upstream process that typically involves:

1. **Querying ChEMBL**: Extracting compounds with known bioactivity (IC50) against ASK1 (Apoptosis signal-regulating kinase 1).
2. **Data Curation**: Standardizing SMILES strings, removing salts, and filtering out invalid molecules.
3. **Descriptor Calculation**: Generating 343 2D descriptors using RDKit for each validated molecule.
4. **Target Enrichment**: Assigning binary classification labels (`Label`) based on an activity threshold and calculating `pIC50` values for regression.

To ensure reproducible research, any proprietary scripts, SQL queries, or Jupyter notebooks used in these preliminary steps should be deposited here.
