import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

def identify_family(col_name):
    if col_name.startswith('WHIM_'): return 'WHIM'
    if col_name.startswith('GETAWAY_'): return 'GETAWAY'
    if col_name.startswith('RDF_'): return 'RDF'
    if col_name.startswith('MORSE_'): return 'MORSE'
    if col_name.startswith('AUTOCORR3D_'): return 'AUTOCORR3D'
    return '2D'

def process_mi_and_corr(X_train_imp, y_train_meta, config):
    mi_limits = config['preprocessing']['mi_top_n']
    corr_thresh = config['preprocessing']['correlation_threshold']
    seed = config['pipeline'].get('global_seed', 42)
    
    # Calculate global Mutual Info
    logging.info("Calculating Mutual Information globally on X_train...")
    mi_scores_reg = mutual_info_regression(X_train_imp, y_train_meta['pIC50'].values, random_state=seed)
    
    if 'Label' in y_train_meta.columns:
        mi_scores_cls = mutual_info_classif(X_train_imp, y_train_meta['Label'].values, random_state=seed)
        mi_scores = (mi_scores_reg + mi_scores_cls) / 2.0
    else:
        mi_scores = mi_scores_reg
        
    mi_series = pd.Series(mi_scores, index=X_train_imp.columns)
    
    families = { "WHIM": [], "GETAWAY": [], "RDF": [], "MORSE": [], "AUTOCORR3D": [], "2D": [] }
    for col in X_train_imp.columns:
        families[identify_family(col)].append(col)
        
    kept_global_cols = []
    
    for fam_name, cols in families.items():
        if not cols: continue
        
        limit = mi_limits.get(fam_name, len(cols))
        fam_mi = mi_series[cols].sort_values(ascending=False)
        top_cols = fam_mi.head(limit).index.tolist()
        
        # Apply Correlation Filter internally to this family
        fam_matrix = X_train_imp[top_cols].corr().abs()
        upper = fam_matrix.where(np.triu(np.ones(fam_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
        
        final_fam = [c for c in top_cols if c not in to_drop]
        kept_global_cols.extend(final_fam)
        logging.info(f"Family {fam_name}: started with {len(cols)}, MI kept top {len(top_cols)}, Correlation kept {len(final_fam)}")
        
    return kept_global_cols

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    logging.basicConfig(filename=config['pipeline']['log_file'], level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - Phase 2 - %(message)s')
    
    logging.info("Loading pre-split extraction artifacts...")
    train_df = pd.read_csv("train_features.csv")
    val_df = pd.read_csv("val_features.csv")
    test_df = pd.read_csv("test_features.csv")
    
    metadata_cols = ['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'pIC50', 'Label', 'pIC50_bin', 'MW_class', 'qed_x_pIC50']
    keep_meta = [c for c in metadata_cols if c in train_df.columns]
    
    y_train_meta = train_df[keep_meta]
    X_train = train_df.drop(columns=keep_meta)
    
    y_val_meta = val_df[keep_meta]
    X_val = val_df.drop(columns=keep_meta)
    
    y_test_meta = test_df[keep_meta]
    X_test = test_df.drop(columns=keep_meta)
    
    # Drop pandas duplicated columns (.1)
    X_train = X_train.loc[:, ~X_train.columns.str.endswith('.1')]
    X_val = X_val.loc[:, ~X_val.columns.str.endswith('.1')]
    X_test = X_test.loc[:, ~X_test.columns.str.endswith('.1')]
    
    non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        X_train = X_train.drop(columns=non_numeric_cols)
        X_val = X_val.drop(columns=non_numeric_cols)
        X_test = X_test.drop(columns=non_numeric_cols)
        
    audit_log = [{"Phase": "Raw", "Train Features": X_train.shape[1], "Val Features": X_val.shape[1], "Test Features": X_test.shape[1]}]

    # 1. Imputation
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)
    audit_log.append({"Phase": "Imputation", "Train Features": X_train_imp.shape[1], "Val Features": X_val_imp.shape[1], "Test Features": X_test_imp.shape[1]})

    # 2. Variance Threshold
    var_thresh = VarianceThreshold(threshold=0.01)
    var_thresh.fit(X_train_imp)
    kept_var_cols = X_train_imp.columns[var_thresh.get_support()]
    
    X_train_var = X_train_imp[kept_var_cols]
    X_val_var = X_val_imp[kept_var_cols]
    X_test_var = X_test_imp[kept_var_cols]
    audit_log.append({"Phase": "Variance Threshold", "Train Features": X_train_var.shape[1], "Val Features": X_val_var.shape[1], "Test Features": X_test_var.shape[1]})

    # 3. MI & Correlation Filtering (Fitted strictly on train target)
    kept_mi_corr_cols = process_mi_and_corr(X_train_var, y_train_meta, config)
    
    X_train_corr = X_train_var[kept_mi_corr_cols]
    X_val_corr = X_val_var[kept_mi_corr_cols]
    X_test_corr = X_test_var[kept_mi_corr_cols]
    
    # Assert validation and test shapes exactly equal training boundary
    assert X_val_corr.shape[1] == X_train_corr.shape[1], "Validation shape mismatch!"
    assert X_test_corr.shape[1] == X_train_corr.shape[1], "Test shape mismatch!"
    audit_log.append({"Phase": "MI + Correlation Filter", "Train Features": X_train_corr.shape[1], "Val Features": X_val_corr.shape[1], "Test Features": X_test_corr.shape[1]})

    # 4. Standard Scaling & Clipping
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(np.clip(scaler.fit_transform(X_train_corr), -5.0, 5.0), columns=X_train_corr.columns)
    X_val_scaled = pd.DataFrame(np.clip(scaler.transform(X_val_corr), -5.0, 5.0), columns=X_train_corr.columns)
    X_test_scaled = pd.DataFrame(np.clip(scaler.transform(X_test_corr), -5.0, 5.0), columns=X_train_corr.columns)
    audit_log.append({"Phase": "Scaling + Clipping", "Train Features": X_train_scaled.shape[1], "Val Features": X_val_scaled.shape[1], "Test Features": X_test_scaled.shape[1]})

    pd.DataFrame(audit_log).to_csv(config['pipeline']['audit_csv'], index=False)
    logging.info("Audit log correctly generated.")
    
    train_final = pd.concat([y_train_meta.reset_index(drop=True), X_train_scaled.reset_index(drop=True)], axis=1)
    val_final = pd.concat([y_val_meta.reset_index(drop=True), X_val_scaled.reset_index(drop=True)], axis=1)
    test_final = pd.concat([y_test_meta.reset_index(drop=True), X_test_scaled.reset_index(drop=True)], axis=1)
    
    train_final.to_csv("train_processed.csv", index=False)
    val_final.to_csv("val_processed.csv", index=False)
    test_final.to_csv("test_processed.csv", index=False)
    
    os.makedirs(config['pipeline']['models_dir'], exist_ok=True)
    joblib.dump(imputer, f"{config['pipeline']['models_dir']}/imputer.pkl")
    joblib.dump(var_thresh, f"{config['pipeline']['models_dir']}/var_thresh.pkl")
    joblib.dump(kept_mi_corr_cols, f"{config['pipeline']['models_dir']}/kept_mi_corr_cols.pkl")
    joblib.dump(scaler, f"{config['pipeline']['models_dir']}/scaler.pkl")
    
    logging.info("Phase 2 Successfully concluded.")

if __name__ == "__main__":
    main()
