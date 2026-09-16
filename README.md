# ASK1 Consensus Inhibitor Prediction Pipeline

This repository contains a reproducible machine-learning pipeline for predicting apoptosis signal-regulating kinase 1 (ASK1) inhibition from molecular descriptors.

The pipeline performs a stratified Murcko-scaffold split, preprocessing and feature selection, Random Forest and XGBoost training, PyTorch ASNN training, and validation with applicability-domain and SHAP analyses.

## Quick start

### 1. Create an environment

Python 3.10 or newer is recommended. Keep the environment outside version control:

```bash
python -m venv .venv
```

Activate it, then install dependencies:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the pipeline

From the repository root:

```bash
python run_pipeline.py
```

The input file and output locations are configured in `config.yaml`. The default input is `phase5_rdkit_2D_descriptors_enriched.csv` and must contain at least `canonical_smiles`, `molecule_chembl_id`, `Label`, and `pIC50`.

### 3. Generate Publication Figures

To generate the 20 black-and-white figures used in the manuscript, run the diagram and plotting scripts after a successful pipeline run:

```bash
python generate_diagrams.py
python generate_data_plots.py
```
All figures will be output to the `publication_figures/` directory.

## Repository layout

- `01_split_and_extract_3d.py` - creates stratified scaffold-based train/validation/test splits.
- `02_preprocessing.py` - imputes, filters, selects, scales, and writes processed features.
- `03_train_models.py` - tunes and trains the ensemble and ASNN models.
- `04_validations.py` - produces evaluation metrics, plots, and SHAP explanations.
- `run_pipeline.py` - runs all four phases in order.
- `models.py` - centralized class architectures (e.g., ASNN).
- `generate_diagrams.py` - programmatic generation of flowchart and schematic architecture figures for publication.
- `generate_data_plots.py` - extraction and generation of all data-driven evaluation plots for publication.
- `generate_3d_pdbs.py` - utility to generate initial 3D conformations via RDKit ETKDG.
- `models/` - trained model artifacts and preprocessing objects.
- `pdb_structures/` - generated molecular structure files.
- `validation_plots/` - validation visualizations.
- `publication_figures/` - directory containing the 20 generated black-and-white publication figures.

The checked-in CSV, model, structure, and plot artifacts make the current project state available to collaborators. Running the pipeline may regenerate them.

## Collaboration

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Do not commit credentials, private datasets, local environments, or unreviewed generated outputs.

## Data and research note

Molecular data and structure files may have their own upstream attribution or usage terms. Check the provenance of any data you add or redistribute. This repository does not replace experimental validation or constitute medical advice.

## License

Source code is released under the MIT License. See [LICENSE](LICENSE). This license does not override terms that may apply to third-party data, models, or generated artifacts.
