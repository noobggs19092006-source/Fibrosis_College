import os
import sys
import yaml
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import (accuracy_score, matthews_corrcoef, roc_auc_score, f1_score, precision_score, recall_score, 
                             confusion_matrix, r2_score, mean_squared_error, mean_absolute_error, balanced_accuracy_score)
from sklearn.base import clone

try:
    import shap
except ImportError:
    logging.error("SHAP is missing.")
    sys.exit("FATAL: SHAP is non-optional. Run: pip install shap")

from models import ASNN

def get_consensus_pred(X, rf, xgb, asnn_model, is_class, threshold=0.5):
    rf_pred = rf.predict_proba(X)[:, 1] if is_class else rf.predict(X)
    xgb_pred = xgb.predict_proba(X)[:, 1] if is_class else xgb.predict(X)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    asnn_model.eval()
    with torch.no_grad():
        nn_pred = asnn_model(X_t).cpu().numpy().flatten()
        
    avg = (rf_pred + xgb_pred + nn_pred) / 3.0
    
    if is_class:
        return (avg >= threshold).astype(int), avg
    return avg

def y_randomization(X_train, y_train, X_test, y_test, rfr_base):
    rand_r2s = []
    y_rand = np.copy(y_train)
    for _ in range(10):
        np.random.shuffle(y_rand)
        rfr_clone = clone(rfr_base)
        rfr_clone.fit(X_train, y_rand)
        pred = rfr_clone.predict(X_test)
        rand_r2s.append(r2_score(y_test, pred))
    return np.mean(rand_r2s)

def applicablity_domain(X_train, X_test, y_test, y_pred):
    pinv = np.linalg.pinv(X_train.T.dot(X_train))
    leverage = np.sum(X_test.dot(pinv) * X_test, axis=1)
    
    residuals = y_test - y_pred
    std_res = residuals / np.std(residuals)
    
    h_star = 3 * (X_train.shape[1] + 1) / X_train.shape[0]
    
    in_ad = np.sum((leverage <= h_star) & (np.abs(std_res) <= 3))
    pct_ad = (in_ad / len(X_test)) * 100.0
    return pct_ad, leverage, std_res, h_star

def do_shap_logic(model, X_test, preds, prefix, config):
    plots_dir = config['pipeline']['plots_dir']
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # SHAP output dimension normalization
    if isinstance(shap_values, list): shap_values = shap_values[1]
    
    plt.figure()
    shap.summary_plot(shap_values, X_test, max_display=20, show=False)
    plt.savefig(f"{plots_dir}/shap_beeswarm_rf_{prefix}.png", bbox_inches='tight')
    plt.close()
    
    top_3_idx = np.argsort(preds)[::-1][:3]
    try:
        if hasattr(explainer, "expected_value"):
            ev = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            for i, idx in enumerate(top_3_idx):
                plt.figure()
                shap.waterfall_plot(shap.Explanation(values=shap_values[idx], base_values=ev, data=X_test[idx]), show=False)
                plt.savefig(f"{plots_dir}/shap_waterfall_rf_{prefix}_Top{i+1}.png", bbox_inches='tight')
                plt.close()
    except Exception as e:
        logging.warning(f"Waterfall plot failed: {e}")

def main():
    with open("config.yaml", "r") as f: config = yaml.safe_load(f)
    logging.basicConfig(filename=config['pipeline']['log_file'], level=logging.INFO)
    plots_dir = config['pipeline']['plots_dir']
    os.makedirs(plots_dir, exist_ok=True)
    
    logging.info("Initializing Phase 4 Publication Validations")
    
    tr = pd.read_csv("train_processed.csv")
    te = pd.read_csv("test_processed.csv")
    
    meta = ['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'pIC50', 'Label', 'pIC50_bin', 'MW_class', 'qed_x_pIC50']
    cols = [c for c in meta if c in tr.columns]
    
    X_train = tr.drop(columns=cols).values.astype(np.float32)
    X_test  = te.drop(columns=cols).values.astype(np.float32)
    
    with open(f"{config['pipeline']['models_dir']}/best_params.json", "r") as f:
        best_p = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    report = []
    
    # --- Classification ---
    if 'Label' in tr.columns:
        yt, yte = tr['Label'].values, te['Label'].values
        rf = joblib.load(f"{config['pipeline']['models_dir']}/rf_cls.pkl")
        xgb = joblib.load(f"{config['pipeline']['models_dir']}/xgb_cls.pkl")
        
        nn_p = best_p['asnn_cls']
        asnn = ASNN(X_train.shape[1], nn_p['nn_drop'], True, tau=config['training']['asnn']['tau_default']).to(device)
        asnn.load_state_dict(torch.load(f"{config['pipeline']['models_dir']}/asnn_cls.pth", map_location=device))
        
        mem = torch.load(f"{config['pipeline']['models_dir']}/asnn_memory_bank_cls.pt", map_location=device)
        asnn.memory_embeddings, asnn.memory_labels = mem['embeddings'], mem['labels']
        asnn.eval_mode_memory_active = True
        
        preds_cls_base, probs_cls = get_consensus_pred(X_test, rf, xgb, asnn, True, threshold=0.5)
        # Apply optimal Youden's J threshold
        thresh_path = f"{config['pipeline']['models_dir']}/optimal_threshold.pkl"
        if os.path.exists(thresh_path):
            opt_thresh = joblib.load(thresh_path)
        else:
            opt_thresh = 0.5
        preds_cls = (probs_cls >= opt_thresh).astype(int)
        logging.info(f"Using optimal threshold: {opt_thresh:.4f}")
        print(f"  Using classification threshold: {opt_thresh:.4f} (Youden's J)")
        
        report.extend([
            {"split": "Test", "model": "Consensus_CLS", "metric": "Accuracy (biased metric — see balanced accuracy)", "value": float(accuracy_score(yte, preds_cls))},
            {"split": "Test", "model": "Consensus_CLS", "metric": "Balanced Accuracy", "value": float(balanced_accuracy_score(yte, preds_cls))},
            {"split": "Test", "model": "Consensus_CLS", "metric": "MCC", "value": float(matthews_corrcoef(yte, preds_cls))},
            {"split": "Test", "model": "Consensus_CLS", "metric": "ROC-AUC", "value": float(roc_auc_score(yte, probs_cls))},
            {"split": "Test", "model": "Consensus_CLS", "metric": "F1-Macro", "value": float(f1_score(yte, preds_cls, average='macro'))},
            {"split": "Test", "model": "Consensus_CLS", "metric": "F1-Weighted", "value": float(f1_score(yte, preds_cls, average='weighted'))},
            {"split": "Test", "model": "Consensus_CLS", "metric": "Precision (Label 0)", "value": float(precision_score(yte, preds_cls, pos_label=0, zero_division=0))},
            {"split": "Test", "model": "Consensus_CLS", "metric": "Recall (Label 0)", "value": float(recall_score(yte, preds_cls, pos_label=0, zero_division=0))},
            {"split": "Test", "model": "Consensus_CLS", "metric": "Precision (Label 1)", "value": float(precision_score(yte, preds_cls, pos_label=1, zero_division=0))},
            {"split": "Test", "model": "Consensus_CLS", "metric": "Recall (Label 1)", "value": float(recall_score(yte, preds_cls, pos_label=1, zero_division=0))}
        ])
        
        plt.figure()
        sns.heatmap(confusion_matrix(yte, preds_cls), annot=True, fmt='d', cmap='Blues')
        plt.savefig(f"{plots_dir}/confusion_matrix.png", bbox_inches='tight')
        plt.close()
        
        do_shap_logic(rf, X_test, probs_cls, "cls", config)

    # --- Regression ---
    if 'pIC50' in tr.columns:
        yt, yte = tr['pIC50'].values, te['pIC50'].values
        rf = joblib.load(f"{config['pipeline']['models_dir']}/rf_reg.pkl")
        xgb = joblib.load(f"{config['pipeline']['models_dir']}/xgb_reg.pkl")
        
        nn_p = best_p['asnn_reg']
        asnn = ASNN(X_train.shape[1], nn_p['nn_drop'], False, tau=config['training']['asnn']['tau_default']).to(device)
        asnn.load_state_dict(torch.load(f"{config['pipeline']['models_dir']}/asnn_reg.pth", map_location=device))
        
        mem = torch.load(f"{config['pipeline']['models_dir']}/asnn_memory_bank_reg.pt", map_location=device)
        asnn.memory_embeddings, asnn.memory_labels = mem['embeddings'], mem['labels']
        asnn.eval_mode_memory_active = True
        
        preds_reg = get_consensus_pred(X_test, rf, xgb, asnn, False)
        
        report.extend([
            {"split": "Test", "model": "Consensus_REG", "metric": "R2", "value": r2_score(yte, preds_reg)},
            {"split": "Test", "model": "Consensus_REG", "metric": "RMSE", "value": np.sqrt(mean_squared_error(yte, preds_reg))},
            {"split": "Test", "model": "Consensus_REG", "metric": "MAE", "value": mean_absolute_error(yte, preds_reg)},
            {"split": "Y-Rand", "model": "RFR_REG", "metric": "Mean_R2", "value": y_randomization(X_train, yt, X_test, yte, rf)}
        ])
        
        ad_pct, _, _, _ = applicablity_domain(X_train, X_test, yte, preds_reg)
        report.append({"split": "Test", "model": "Consensus_REG", "metric": "AD_Inside_Pct", "value": ad_pct})
        
        plt.figure()
        plt.scatter(yte, preds_reg, alpha=0.6)
        plt.plot([min(yte), max(yte)], [min(yte), max(yte)], 'r--')
        plt.xlabel('Actual pIC50')
        plt.ylabel('Predicted pIC50')
        plt.title('Regression Scatter Plot')
        plt.savefig(f"{plots_dir}/reg_scatter.png", bbox_inches='tight')
        plt.close()
        
        do_shap_logic(rf, X_test, preds_reg, "reg", config)

    with open("validation_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    logging.info("Phase 4 Completed. Report safely cached.")

if __name__ == "__main__":
    main()
