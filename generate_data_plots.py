import os
import re
import yaml
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (roc_curve, auc, confusion_matrix, precision_recall_curve, 
                             r2_score, accuracy_score, matthews_corrcoef, balanced_accuracy_score, roc_auc_score)
from sklearn.feature_selection import mutual_info_regression
from sklearn.base import clone
import shap
from rdkit import Chem
from rdkit.Chem import Draw
from models import ASNN

# Formatting for black and white
plt.style.use('grayscale')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['black', 'dimgray', 'gray', 'silver'])
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['image.cmap'] = 'Greys'

out_dir = "publication_figures"
os.makedirs(out_dir, exist_ok=True)

with open("config.yaml", "r") as f: 
    config = yaml.safe_load(f)

models_dir = config['pipeline']['models_dir']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    tr = pd.read_csv("train_processed.csv")
    te = pd.read_csv("test_processed.csv")
    meta = ['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'pIC50', 'Label', 'pIC50_bin', 'MW_class', 'qed_x_pIC50']
    cols = [c for c in meta if c in tr.columns]
    
    X_train = tr.drop(columns=cols).values.astype(np.float32)
    X_test  = te.drop(columns=cols).values.astype(np.float32)
    
    return tr, te, X_train, X_test, cols

def get_consensus_pred(X, rf, xgb, asnn_model, is_class, threshold=0.5):
    rf_pred = rf.predict_proba(X)[:, 1] if is_class else rf.predict(X)
    xgb_pred = xgb.predict_proba(X)[:, 1] if is_class else xgb.predict(X)
    
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    asnn_model.eval()
    with torch.no_grad():
        nn_pred = asnn_model(X_t).cpu().numpy().flatten()
        
    avg = (rf_pred + xgb_pred + nn_pred) / 3.0
    if is_class:
        return (avg >= threshold).astype(int), avg, rf_pred, xgb_pred, nn_pred
    return avg, rf_pred, xgb_pred, nn_pred

def fig5_optuna_history():
    print("Figure 5: Optuna History")
    log_file = config['pipeline']['log_file']
    trials = []
    scores = []
    with open(log_file, "r") as f:
        for line in f:
            m = re.search(r"Trial (\d+) finished with value: ([\-\d\.]+)", line)
            if m:
                trials.append(int(m.group(1)))
                scores.append(float(m.group(2)))
    
    if trials:
        plt.figure(figsize=(6, 4))
        plt.scatter(trials, scores, color='black', alpha=0.6)
        plt.plot(trials, [max(scores[:i+1]) if scores[0] > 0 else min(scores[:i+1]) for i in range(len(scores))], 
                 color='dimgray', linestyle='--', label='Best Value')
        plt.xlabel("Trials")
        plt.ylabel("Validation Score")
        plt.title("Hyperparameter Optimization Convergence")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/Figure_5_Optuna_History.png", dpi=300)
        plt.close()

def classification_figures(tr, te, X_train, X_test):
    print("Generating Classification Figures (6, 7, 8, 20)")
    y_test = te['Label'].values
    
    rf = joblib.load(f"{models_dir}/rf_cls.pkl")
    xgb = joblib.load(f"{models_dir}/xgb_cls.pkl")
    with open(f"{models_dir}/best_params.json", "r") as f: best_p = json.load(f)
    
    nn_p = best_p['asnn_cls']
    asnn = ASNN(X_train.shape[1], nn_p['nn_drop'], True, tau=config['training']['asnn']['tau_default']).to(device)
    asnn.load_state_dict(torch.load(f"{models_dir}/asnn_cls.pth", map_location=device))
    mem = torch.load(f"{models_dir}/asnn_memory_bank_cls.pt", map_location=device)
    asnn.memory_embeddings, asnn.memory_labels = mem['embeddings'], mem['labels']
    asnn.eval_mode_memory_active = True
    
    opt_thresh = 0.5
    thresh_path = f"{models_dir}/optimal_threshold.pkl"
    if os.path.exists(thresh_path):
        opt_thresh = joblib.load(thresh_path)
        
    preds_cls, probs_cls, rf_prob, xgb_prob, nn_prob = get_consensus_pred(X_test, rf, xgb, asnn, True, threshold=opt_thresh)
    
    # Figure 6: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs_cls)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='black', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_6_ROC_Curve.png", dpi=300)
    plt.close()
    
    # Figure 7: Confusion Matrix
    cm = confusion_matrix(y_test, preds_cls)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', cbar=False,
                xticklabels=['Inactive', 'Active'], yticklabels=['Inactive', 'Active'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_7_Confusion_Matrix.png", dpi=300)
    plt.close()
    
    # Figure 8: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, probs_cls)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, color='black', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_8_PR_Curve.png", dpi=300)
    plt.close()
    
    # Figure 20: Model Comparison
    models = ['RF', 'XGBoost', 'ASNN', 'Consensus']
    probs = [rf_prob, xgb_prob, nn_prob, probs_cls]
    
    auc_scores = [roc_auc_score(y_test, p) for p in probs]
    mcc_scores = [matthews_corrcoef(y_test, (p >= opt_thresh).astype(int)) for p in probs]
    bacc_scores = [balanced_accuracy_score(y_test, (p >= opt_thresh).astype(int)) for p in probs]
    
    x = np.arange(len(models))
    width = 0.25
    plt.figure(figsize=(8, 5))
    plt.bar(x - width, auc_scores, width, label='ROC-AUC', color='dimgray', edgecolor='black')
    plt.bar(x, bacc_scores, width, label='Balanced Acc', color='silver', edgecolor='black')
    plt.bar(x + width, mcc_scores, width, label='MCC', color='white', edgecolor='black', hatch='//')
    
    plt.xticks(x, models)
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_20_Model_Comparison.png", dpi=300)
    plt.close()

def regression_figures(tr, te, X_train, X_test):
    print("Generating Regression Figures (9, 10, 11, 12, 13, 14)")
    y_train = tr['pIC50'].values
    y_test = te['pIC50'].values
    
    rf = joblib.load(f"{models_dir}/rf_reg.pkl")
    xgb = joblib.load(f"{models_dir}/xgb_reg.pkl")
    with open(f"{models_dir}/best_params.json", "r") as f: best_p = json.load(f)
    
    nn_p = best_p['asnn_reg']
    asnn = ASNN(X_train.shape[1], nn_p['nn_drop'], False, tau=config['training']['asnn']['tau_default']).to(device)
    asnn.load_state_dict(torch.load(f"{models_dir}/asnn_reg.pth", map_location=device))
    mem = torch.load(f"{models_dir}/asnn_memory_bank_reg.pt", map_location=device)
    asnn.memory_embeddings, asnn.memory_labels = mem['embeddings'], mem['labels']
    asnn.eval_mode_memory_active = True
    
    preds_reg, _, _, _ = get_consensus_pred(X_test, rf, xgb, asnn, False)
    
    # Figure 9: Pred vs Exp
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, preds_reg, color='black', alpha=0.6, edgecolors='none')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
    plt.xlabel('Experimental pIC50')
    plt.ylabel('Predicted pIC50')
    plt.title('Predicted vs Experimental (Consensus)')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_9_Pred_vs_Exp.png", dpi=300)
    plt.close()
    
    # Figure 10: Residual Plot
    residuals = y_test - preds_reg
    plt.figure(figsize=(6, 4))
    plt.scatter(preds_reg, residuals, color='black', alpha=0.6, edgecolors='none')
    plt.axhline(0, color='gray', linestyle='--', lw=2)
    plt.xlabel('Predicted pIC50')
    plt.ylabel('Residual (True - Pred)')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_10_Residual.png", dpi=300)
    plt.close()
    
    # SHAP (11, 12)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    # Beeswarm (Greyscale workaround for SHAP)
    plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, X_test, max_display=15, show=False, cmap=plt.get_cmap("gray"))
    plt.title('SHAP Feature Importance (RF component)')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_11_SHAP_Beeswarm.png", dpi=300)
    plt.close()
    
    # Bar plot
    plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, X_test, max_display=20, plot_type='bar', show=False, color='dimgray')
    plt.title('Top 20 Features (SHAP)')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_12_SHAP_Bar.png", dpi=300)
    plt.close()
    
    # Figure 13: AD (Williams Plot)
    pinv = np.linalg.pinv(X_train.T.dot(X_train))
    leverage = np.sum(X_test.dot(pinv) * X_test, axis=1)
    std_res = residuals / np.std(residuals)
    h_star = 3 * (X_train.shape[1] + 1) / X_train.shape[0]
    
    plt.figure(figsize=(6, 5))
    plt.scatter(leverage, std_res, color='black', alpha=0.6)
    plt.axhline(3, color='gray', linestyle='--')
    plt.axhline(-3, color='gray', linestyle='--')
    plt.axvline(h_star, color='gray', linestyle='--')
    plt.xlabel('Leverage')
    plt.ylabel('Standardized Residuals')
    plt.title('Williams Plot (Applicability Domain)')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_13_Williams_Plot.png", dpi=300)
    plt.close()
    
    # Figure 14: Y-Randomization
    print("Running Y-Randomization...")
    rand_r2s = []
    y_rand = np.copy(y_train)
    for _ in range(10):
        np.random.shuffle(y_rand)
        rfr_clone = clone(rf)
        rfr_clone.fit(X_train, y_rand)
        pred = rfr_clone.predict(X_test)
        rand_r2s.append(r2_score(y_test, pred))
        
    true_r2 = r2_score(y_test, preds_reg)
    
    plt.figure(figsize=(5, 5))
    plt.scatter([1]*10, rand_r2s, color='gray', label='Y-Scrambled (10 runs)')
    plt.scatter([2], [true_r2], color='black', s=100, marker='*', label='Original Model')
    plt.xlim(0, 3)
    plt.xticks([1, 2], ['Scrambled', 'Original'])
    plt.ylabel('Test R² Score')
    plt.title('Y-Randomization Test')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_14_Y_Randomization.png", dpi=300)
    plt.close()

def extra_figures(tr, te, X_train, cols):
    print("Generating Extra Figures (15, 16, 17, 18, 19)")
    
    # Figure 15: Chem Structure
    smiles = tr[tr['molecule_chembl_id'] == 'CHEMBL3925596']['canonical_smiles'].values
    if len(smiles) > 0:
        mol = Chem.MolFromSmiles(smiles[0])
        opts = Draw.MolDrawOptions()
        opts.useBWAtomPalette()
        Draw.MolToFile(mol, f"{out_dir}/Figure_15_CHEMBL3925596.png", size=(400, 400), options=opts)
    
    # Figure 16: Correlation Heatmap
    # We will compute corr on the first 30 features just to make it readable
    corr = pd.DataFrame(X_train[:, :30]).corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, cmap='Greys', cbar=True, xticklabels=False, yticklabels=False)
    plt.title('Feature Correlation Heatmap (Subset)')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_16_Correlation.png", dpi=300)
    plt.close()
    
    # Figure 17: MI Distribution
    print("Calculating MI for Figure 17...")
    y_train = tr['pIC50'].values
    seed = config['pipeline'].get('global_seed', 42)
    mi = mutual_info_regression(X_train, y_train, random_state=seed)
    top_idx = np.argsort(mi)[::-1][:20]
    
    plt.figure(figsize=(7, 5))
    plt.bar(range(20), mi[top_idx], color='dimgray', edgecolor='black')
    plt.xlabel('Top 20 Features (Ranked)')
    plt.ylabel('Mutual Information Score')
    plt.title('Feature Importance (MI)')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_17_MI_Bar.png", dpi=300)
    plt.close()
    
    # Figure 18: Dataset Distribution (pIC50)
    plt.figure(figsize=(5, 4))
    plt.hist(tr['pIC50'], bins=20, color='silver', edgecolor='black')
    plt.xlabel('pIC50')
    plt.ylabel('Frequency')
    plt.title('pIC50 Distribution (Train)')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Figure_18_pIC50_Dist.png", dpi=300)
    plt.close()
    
    # Figure 19: Class Distribution
    if 'Label' in tr.columns:
        counts = tr['Label'].value_counts()
        plt.figure(figsize=(4, 5))
        plt.bar([0, 1], [counts.get(0, 0), counts.get(1, 0)], color=['silver', 'dimgray'], edgecolor='black')
        plt.xticks([0, 1], ['Inactive (0)', 'Active (1)'])
        plt.ylabel('Count')
        plt.title('Class Imbalance')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/Figure_19_Class_Dist.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    tr, te, X_train, X_test, cols = load_data()
    fig5_optuna_history()
    
    if 'Label' in tr.columns:
        classification_figures(tr, te, X_train, X_test)
        
    if 'pIC50' in tr.columns:
        regression_figures(tr, te, X_train, X_test)
        
    extra_figures(tr, te, X_train, cols)
    print("Done generating data plots!")
