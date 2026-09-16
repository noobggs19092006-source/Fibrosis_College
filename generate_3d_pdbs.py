"""
=============================================================================
  3D PDB Generator for Fibrosis Project
  Method  : RDKit ETKDG conformer generation
  Input   : phase5_rdkit_2D_descriptors_enriched.csv  (or original CSV)
  Output  : ./pdb_structures/<molecule_chembl_id>.pdb  (one file per molecule)
=============================================================================
  Requirements:
      pip install rdkit pandas tqdm

  Usage:
      python generate_3d_pdbs.py

  Optional flags (edit CONFIG below):
      - ENERGY_MINIMIZE : set True to add MMFF94 minimization (slower but better)
      - NUM_CONFORMERS  : number of conformers to try before picking best
      - RANDOM_SEED     : for reproducibility
=============================================================================
"""

import os
import sys
import pandas as pd
from pathlib import Path

# ── Try importing required libraries ─────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')   # suppress RDKit warnings
except ImportError:
    print("❌  RDKit not found. Install it with:  pip install rdkit")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("⚠️  tqdm not found (pip install tqdm). Using plain progress counter.")
    tqdm = None

import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these as needed
# ══════════════════════════════════════════════════════════════════════════════
CSV_PATH        = "phase5_rdkit_2D_descriptors_enriched.csv"   # path to your CSV
OUTPUT_DIR      = "./pdb_structures"  # relative path for cross-platform compatibility
ENERGY_MINIMIZE = False   # True = MMFF94 minimization (recommended but slower)
NUM_CONFORMERS  = 1       # 1 is fine for ETKDG; increase to 5 for better sampling
RANDOM_SEED     = config['pipeline'].get('global_seed', 42)
ADD_HYDROGENS   = True    # include explicit H atoms in PDB (recommended for VMD)
# ══════════════════════════════════════════════════════════════════════════════


def smiles_to_pdb(smiles: str, mol_id: str, output_path: str) -> tuple[bool, str]:
    """
    Convert a SMILES string to a 3D PDB file using RDKit ETKDG.

    Returns:
        (success: bool, message: str)
    """
    # 1. Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, f"Invalid SMILES — could not parse molecule"

    # 2. Add hydrogens
    if ADD_HYDROGENS:
        mol = Chem.AddHs(mol)

    # 3. Embed 3D coordinates using ETKDG
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    params.numThreads = 0          # use all available CPU threads
    params.enforceChirality = True

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=NUM_CONFORMERS, params=params)

    if len(conf_ids) == 0:
        # Fallback: try without chirality enforcement
        params.enforceChirality = False
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=NUM_CONFORMERS, params=params)

    if len(conf_ids) == 0:
        return False, "ETKDG embedding failed — could not generate 3D coordinates"

    # 4. Optional: MMFF94 energy minimization
    if ENERGY_MINIMIZE:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94")
        # Pick lowest energy conformer
        if results:
            energies = [(r[1], i) for i, r in enumerate(results) if r[0] == 0]
            if energies:
                best_conf = min(energies)[1]
            else:
                best_conf = 0
        else:
            best_conf = 0
    else:
        best_conf = 0  # use first conformer from ETKDG

    # 5. Write PDB
    try:
        writer = Chem.PDBWriter(output_path)
        writer.write(mol, confId=int(conf_ids[best_conf]))
        writer.close()
        return True, "OK"
    except Exception as e:
        return False, f"PDB write error: {e}"


def main():
    # ── Load dataset ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Fibrosis 3D Structure Generator")
    print(f"{'='*60}")

    if not os.path.exists(CSV_PATH):
        print(f"❌  CSV not found: {CSV_PATH}")
        print("    Make sure the script is in the same folder as your CSV.")
        sys.exit(1)

    print(f"\n📂  Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    required_cols = {'molecule_chembl_id', 'canonical_smiles'}
    if not required_cols.issubset(df.columns):
        print(f"❌  CSV must contain columns: {required_cols}")
        sys.exit(1)

    print(f"✅  Loaded {len(df):,} compounds")

    # ── Create output directory ───────────────────────────────────────────────
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁  Output folder: {out_dir.resolve()}\n")

    # ── Generate 3D PDB files ─────────────────────────────────────────────────
    success_count = 0
    fail_count    = 0
    failed_mols   = []

    iterator = df.iterrows()
    if tqdm:
        iterator = tqdm(df.iterrows(), total=len(df), desc="Generating PDBs",
                        unit="mol", ncols=80, colour="green")

    for idx, row in iterator:
        mol_id = str(row['molecule_chembl_id']).strip()
        smiles = str(row['canonical_smiles']).strip()
        pdb_path = str(out_dir / f"{mol_id}.pdb")

        # Skip if already exists (allows resuming interrupted runs)
        if os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 0:
            success_count += 1
            continue

        ok, msg = smiles_to_pdb(smiles, mol_id, pdb_path)

        if ok:
            success_count += 1
        else:
            fail_count += 1
            failed_mols.append({'molecule_chembl_id': mol_id,
                                 'canonical_smiles': smiles,
                                 'error': msg})
            if tqdm is None:
                print(f"  ⚠️  [{idx+1}/{len(df)}] FAILED {mol_id}: {msg}")

        # Plain progress counter if tqdm not available
        if tqdm is None and (idx + 1) % 100 == 0:
            print(f"  Progress: {idx+1}/{len(df)} | ✅ {success_count} | ❌ {fail_count}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✅  Successfully generated : {success_count:,} PDB files")
    print(f"  ❌  Failed                 : {fail_count:,} molecules")
    print(f"  📁  Saved to               : {out_dir.resolve()}")
    print(f"{'='*60}")

    # Save failed molecules log
    if failed_mols:
        fail_log = "failed_molecules.csv"
        pd.DataFrame(failed_mols).to_csv(fail_log, index=False)
        print(f"\n⚠️   Failed molecules logged to: {fail_log}")
        print("    Common causes: invalid SMILES, very large/complex molecules,")
        print("    disconnected fragments, or rare atom types.")

    print("\n🔬  To load all structures in VMD:")
    print(f"    1. Open VMD → File → New Molecule")
    print(f"    2. Browse to any .pdb in:  {out_dir.resolve()}")
    print(f"    3. Or use the TCL console to batch load (see vmd_batch_load.tcl)")
    print()


if __name__ == "__main__":
    main()
