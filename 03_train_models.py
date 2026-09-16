import os
import sys
import yaml
import json
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, balanced_accuracy_score
from sklearn.metrics import roc_curve

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    logging.error("XGBoost is strictly required. pip install xgboost")
    sys.exit("FATAL: xgboost not importable.")

try:
    from imblearn.combine import SMOTETomek
except ImportError:
    SMOTETomek = None

from models import ASNN

def train_asnn(X_train, y_train, X_val, y_val, is_class, config, lr, dropout):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ASNN(X_train.shape[1], dropout, is_class, tau=config['training']['asnn']['tau_default']).to(device)
    
    criterion = nn.BCEWithLogitsLoss() if is_class else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    tx, ty = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    vx, vy = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    
    loader = DataLoader(TensorDataset(tx, ty), batch_size=32, shuffle=True)
    
    best_val_loss = float('inf')
    patience = config['training']['early_stopping_patience']
    patience_counter = 0
    best_state = None
    
    history_train, history_val = [], []
    
    for epoch in range(config['training']['max_epochs']):
        model.train()
        train_losses = []
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        model.eval()
        with torch.no_grad():
            val_out = model(vx)
            val_loss = criterion(val_out, vy).item()
            
        history_train.append(np.mean(train_losses))
        history_val.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    model.load_state_dict(best_state)
    return model, best_val_loss, history_train, history_val

def build_datasets(is_class, seed):
    tr = pd.read_csv("train_processed.csv")
    va = pd.read_csv("val_processed.csv")
    
    meta = ['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'pIC50', 'Label', 'pIC50_bin', 'MW_class', 'qed_x_pIC50']
    cols = [c for c in meta if c in tr.columns]
    
    Xt, yt = tr.drop(columns=cols).values.astype(np.float32), tr['Label'].values if is_class else tr['pIC50'].values
    Xv, yv = va.drop(columns=cols).values.astype(np.float32), va['Label'].values if is_class else va['pIC50'].values
    
    if is_class and SMOTETomek is not None:
        Xt, yt = SMOTETomek(random_state=seed).fit_resample(Xt, yt)
        
    return Xt, yt, Xv, yv

def main():
    with open("config.yaml", "r") as f: config = yaml.safe_load(f)
    logging.basicConfig(filename=config['pipeline']['log_file'], level=logging.INFO)
    logging.info("Initializing Phase 3 Models Tuning & ASNN generation")
    
    seed = config['pipeline'].get('global_seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.makedirs(config['pipeline']['models_dir'], exist_ok=True)
    best_params = {}
    
    for is_class, prefix in [(True, "cls"), (False, "reg")]:
        logging.info(f"--- Training {prefix.upper()} Logic ---")
        Xt, yt, Xv, yv = list(build_datasets(is_class, seed))
        
        # 1. OPTUNA TUNING
        def objective(trial):
            rf_n = trial.suggest_int('rf_n', 50, 200)
            rf_d = trial.suggest_int('rf_d', 5, 20)
            rf_l = trial.suggest_int('rf_l', 1, 5)
            
            xgb_n = trial.suggest_int('xgb_n', 50, 200)
            xgb_d = trial.suggest_int('xgb_d', 3, 10)
            xgb_lr = trial.suggest_float('xgb_lr', 1e-3, 0.3, log=True)
            xgb_sub = trial.suggest_float('xgb_sub', 0.5, 1.0)
            
            if is_class:
                rf = RandomForestClassifier(n_estimators=rf_n, max_depth=rf_d, min_samples_leaf=rf_l, random_state=seed, n_jobs=-1, class_weight='balanced')
                rf.fit(Xt, yt)
                # Optimize for balanced accuracy — correct objective for 22:1 imbalance
                rf_score = balanced_accuracy_score(yv, rf.predict(Xv))
                
                pos_weight = float(np.sum(yt==0))/np.sum(yt==1) if np.sum(yt==1)>0 else 1.0
                xgb = XGBClassifier(n_estimators=xgb_n, max_depth=xgb_d, learning_rate=xgb_lr, subsample=xgb_sub, random_state=seed, scale_pos_weight=pos_weight)
                xgb.fit(Xt, yt)
                xgb_score = balanced_accuracy_score(yv, xgb.predict(Xv))
                
                return (rf_score + xgb_score) / 2.0
            else:
                rf = RandomForestRegressor(n_estimators=rf_n, max_depth=rf_d, min_samples_leaf=rf_l, random_state=seed, n_jobs=-1)
                rf.fit(Xt, yt)
                rf_score = -mean_squared_error(yv, rf.predict(Xv))  # negative for maximization
                
                xgb = XGBRegressor(n_estimators=xgb_n, max_depth=xgb_d, learning_rate=xgb_lr, subsample=xgb_sub, random_state=seed)
                xgb.fit(Xt, yt)
                xgb_score = -mean_squared_error(yv, xgb.predict(Xv))
                
                return (rf_score + xgb_score) / 2.0
            
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=config['training']['optuna_trials'])
        
        p = study.best_params
        best_params[prefix] = p
        
        # 2. FINAL TRAINING XGB & RF
        pos_weight = float(np.sum(yt==0))/np.sum(yt==1) if is_class and np.sum(yt==1)>0 else 1.0
        
        if is_class:
            rf = RandomForestClassifier(n_estimators=p['rf_n'], max_depth=p['rf_d'], min_samples_leaf=p['rf_l'], random_state=seed, n_jobs=-1, class_weight='balanced')
            xgb = XGBClassifier(n_estimators=p['xgb_n'], max_depth=p['xgb_d'], learning_rate=p['xgb_lr'], subsample=p['xgb_sub'], random_state=seed, scale_pos_weight=pos_weight)
        else:
            rf = RandomForestRegressor(n_estimators=p['rf_n'], max_depth=p['rf_d'], min_samples_leaf=p['rf_l'], random_state=seed, n_jobs=-1)
            xgb = XGBRegressor(n_estimators=p['xgb_n'], max_depth=p['xgb_d'], learning_rate=p['xgb_lr'], subsample=p['xgb_sub'], random_state=seed)
            
        rf.fit(Xt, yt)
        xgb.fit(Xt, yt)
        
        import joblib
        joblib.dump(rf, f"{config['pipeline']['models_dir']}/rf_{prefix}.pkl")
        joblib.dump(xgb, f"{config['pipeline']['models_dir']}/xgb_{prefix}.pkl")
        
        # Find optimal classification threshold using Youden's J on validation set
        if is_class:
            rf_vp  = rf.predict_proba(Xv)[:, 1]
            xgb_vp = xgb.predict_proba(Xv)[:, 1]
            avg_vp = (rf_vp + xgb_vp) / 2.0
            fpr, tpr, thresholds = roc_curve(yv, avg_vp)
            youden_j = tpr - fpr
            best_thresh = float(thresholds[np.argmax(youden_j)])
            best_thresh = max(0.1, min(0.9, best_thresh))  # clip to safe range
            joblib.dump(best_thresh, f"{config['pipeline']['models_dir']}/optimal_threshold.pkl")
            logging.info(f"Optimal classification threshold (Youden's J): {best_thresh:.4f}")
            print(f"  Optimal threshold (Youden's J on val set): {best_thresh:.4f}")
        
        # 3. ASNN PYTORCH
        def asnn_obj(trial):
            lr = trial.suggest_float('nn_lr', 1e-4, 1e-2, log=True)
            dropout = trial.suggest_float('nn_drop', 0.1, 0.5)
            _, val_loss, _, _ = train_asnn(Xt, yt, Xv, yv, is_class, config, lr, dropout)
            return val_loss
            
        study_nn = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study_nn.optimize(asnn_obj, n_trials=10) # 10 trials for NN specific
        best_nn_p = study_nn.best_params
        best_params[f"asnn_{prefix}"] = best_nn_p
        
        model_asnn, _, t_loss, v_loss = train_asnn(Xt, yt, Xv, yv, is_class, config, best_nn_p['nn_lr'], best_nn_p['nn_drop'])
        
        # Populate ASNN memory
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_asnn.populate_memory(torch.tensor(Xt, dtype=torch.float32).to(device), torch.tensor(yt, dtype=torch.float32).unsqueeze(1).to(device))
        
        torch.save(model_asnn.state_dict(), f"{config['pipeline']['models_dir']}/asnn_{prefix}.pth")
        torch.save({'embeddings': model_asnn.memory_embeddings, 'labels': model_asnn.memory_labels}, f"{config['pipeline']['models_dir']}/asnn_memory_bank_{prefix}.pt")
        
        # Plot curves
        plt.figure()
        plt.plot(t_loss, label='Train Loss')
        plt.plot(v_loss, label='Val Loss')
        plt.legend()
        plt.title(f"{prefix.upper()} ASNN Learning Curves")
        plt.savefig(f"{config['pipeline']['models_dir']}/training_curves_{prefix}.png")
        plt.close()
        
    with open(f"{config['pipeline']['models_dir']}/best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
        
    logging.info("Phase 3 Successfully concluded.")

if __name__ == "__main__":
    main()
