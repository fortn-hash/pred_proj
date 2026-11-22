from pathlib import Path
import json
import random
import joblib
import numpy as np
import pandas as pd
import optuna
from scipy.special import logit
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

seed=1234
random.seed(1234)
np.random.seed(1234)
cols = ['Age', 'Sex', 'LDH', 'PBBABS', 'BMB']
names = ["LR", 'KNN', 'SVM', 'DT', 'RF', 'XGBoost', 'LightGBM', 'MLP']

def search(t, name):
    if name == 'LR':
        return dict(C=t.suggest_float('C', 0.001, 100, log=True),
                    penalty=t.suggest_categorical('penalty', ['l1', 'l2']))
    if name == 'KNN':
        return dict(n_neighbors=t.suggest_int('n_neighbors', 3, 100),
                    weights=t.suggest_categorical('weights', ['uniform', 'distance']),
                    p=t.suggest_int('p', 1, 2))
    if name == 'SVM':
        p = dict(kernel=t.suggest_categorical('kernel', ['linear', 'rbf']),
                 C=t.suggest_float('C', 0.001, 1000, log=True),
                 class_weight=t.suggest_categorical('class_weight', [None, 'balanced']))
        if p['kernel'] == 'rbf':
            p['gamma'] = t.suggest_float('gamma', 0.0001, 1, log=True)
        return p
    if name == 'DT':
        return dict(max_depth=t.suggest_int('max_depth', 2, 6),
                    min_samples_leaf=t.suggest_int('min_samples_leaf', 5, 60),
                    ccp_alpha=t.suggest_categorical('ccp_alpha', [0, 0.001, 0.005, 0.01, 0.02]),
                    criterion=t.suggest_categorical('criterion', ['gini', 'entropy']))
    if name == 'RF':
        return dict(n_estimators=t.suggest_int('n_estimators', 200, 600, step=100),
                    max_depth=t.suggest_int('max_depth', 3, 12),
                    min_samples_leaf=t.suggest_int('min_samples_leaf', 1, 15),
                    max_features=t.suggest_int('max_features', 1, len(cols)),
                    max_samples=t.suggest_float('max_samples', 0.6, 1.0, step=0.001),
                    criterion=t.suggest_categorical('criterion', ['gini', 'entropy']))
    if name in ['XGBoost', 'LightGBM']:
        p=dict(n_estimators=t.suggest_int('n_estimators', 100, 1000, step=50),
                 learning_rate=t.suggest_float('learning_rate', 0.005, 0.2, log=True),
                 subsample=t.suggest_float('subsample', 0.5, 1),
                 colsample_bytree=t.suggest_float('colsample_bytree', 0.5, 1),
                 reg_lambda=t.suggest_float('reg_lambda', 0.01, 100, log=True))
        if name == 'XGBoost':
            p.update(max_depth=t.suggest_int('max_depth', 2, 5),
                     min_child_weight=t.suggest_float('min_child_weight', 1, 20, log=True),
                     gamma=t.suggest_float('gamma', 0, 5))
        else:
            p.update(num_leaves=t.suggest_int('num_leaves', 3, 15),
                     min_child_samples=t.suggest_int('min_child_samples', 10, 50))
        return p
    p = dict(hidden=t.suggest_categorical('hidden', ['4', '8', '16', '32', '16,8']),
             alpha=t.suggest_float('alpha', 0.0001, 100, log=True),
             solver=t.suggest_categorical('solver', ['adam', 'lbfgs']))
    if p['solver'] == 'adam':
        p['learning_rate_init'] = t.suggest_float('learning_rate_init', 0.0001, 0.01, log=True)
    return p
def model(name, p):
    p=(p.copy())
    if name == 'LR':
        solver = 'liblinear' if p['penalty'] == 'l1' else 'lbfgs'
        a = LogisticRegression(**p, solver=solver, max_iter=5000, random_state=seed)
    elif name == 'KNN':
        a = KNeighborsClassifier(**p, n_jobs=1)
    elif name == 'SVM':
        a = SVC(**p, probability=False, random_state=seed)
    elif name == 'DT':
        a = DecisionTreeClassifier(**p, random_state=seed)
    elif name == 'RF':
        a = RandomForestClassifier(**p, random_state=seed, n_jobs=1)
    elif name == 'XGBoost':
        a = XGBClassifier(**p, objective='binary:logistic', eval_metric='logloss',
                          tree_method='hist', random_state=seed, n_jobs=1)
    elif name == 'LightGBM':
        a = LGBMClassifier(**p, objective='binary', subsample_freq=1, verbosity=-1,
                           deterministic=True, force_col_wise=True, random_state=seed, n_jobs=1)
    else:
        h = tuple(int(v) for v in p.pop('hidden').split(','))
        a = MLPClassifier(**p, hidden_layer_sizes=h, max_iter=3000,
                          early_stopping=False, random_state=seed)
    if name in ['LR', 'KNN', 'SVM', 'MLP']:
        a = make_pipeline(StandardScaler(), a)
    return a

def raw(fit, x, name):
    return fit.decision_function(x) if name == 'SVM' else fit.predict_proba(x)[:, 1]
def apply_platt(a, p, name):
    if a is None: return p
    z = p.reshape(-1, 1) if name == 'SVM' else logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return a.predict_proba(z)[:, 1]
def metrics(y, p, cut):
    tn, fp, fn, tp = confusion_matrix(y, p >= cut, labels=[0, 1]).ravel()
    div = lambda a, b: a / b if b else np.nan
    rng=np.random.default_rng(seed)
    neg, pos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    auc = []
    for b in range(2000):
        idx = np.r_[rng.choice(neg, len(neg), replace=True),
                    rng.choice(pos, len(pos), replace=True)]
        auc.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.quantile(
              auc, [0.025, 0.975])
    return dict(AUC=roc_auc_score(y, p), lower=lo, upper=hi,
                sensitivity=div(tp, tp + fn), specificity=div(tn, tn + fp),
                accuracy=div(tp + tn, len(y)), PPV=div(tp, tp + fp),
                NPV=div(tn, tn + fn), F1=div(2 * tp, 2 * tp + fp + fn),
                Brier=brier_score_loss(y, p), threshold=float(cut),
                TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))
def main():
    dat = [pd.read_csv(f'processed/train{i}.csv') for i in range(1, 11)]
    dat1 = [pd.read_csv(f"processed/validation{i}.csv")
                 for i in range(1,11)]
    x = [df[cols].to_numpy(dtype=float) for df in dat]
    x_valid = [df[cols].to_numpy(dtype=float) for df in dat1]
    y = dat[0]['IDH'].to_numpy(dtype=int)
    y_valid = dat1[0]['IDH'].to_numpy(dtype=int)
    Path('model').mkdir(exist_ok=True)
    Path('result').mkdir(exist_ok=True)
    # the same fold
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    folds = list(cv.split(x[0], y))
    out = dat[0][['SampleID', 'IDH']].copy()
    out1 = dat1[0][['SampleID', 'IDH']].copy()
    out2 = out.copy()
    res, params = [], {}
    for name in names:
        print(name, flush=True)
        def objective(t):
            p = search(t, name)
            score = []
            for i in range(2):
                for train_idx, valid_idx in folds:
                    fit = model(name, p)
                    fit.fit(x[i][train_idx], y[train_idx])
                    prob = raw(fit, x[i][valid_idx], name)
                    score.append(roc_auc_score(y[valid_idx], prob))
            return float(np.mean(score))
        study = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=100, n_jobs=1)
        p = study.best_params
        params[name] = p
        study.trials_dataframe().to_csv(f'result/{name}.csv', index=False)
        oof = np.zeros(len(y))
        cnt = np.zeros(len(y))
        for train_idx, valid_idx in folds:
            prob = np.zeros(len(valid_idx))
            for i in range(10):
                fit = model(name, p)
                fit.fit(x[i][train_idx], y[train_idx])
                prob += raw(fit, x[i][valid_idx], name) / 10
            oof[valid_idx] += prob
            cnt[valid_idx] += 1
        oof /= cnt
        # Platt 
        cal = None
        oof_cal = oof.copy()
        if name not in ['LR', 'DT']:
            z = oof.reshape(-1, 1) if name == 'SVM' else logit(np.clip(oof, 1e-6, 1 - 1e-6)).reshape(-1, 1)
            cal=LogisticRegression(C=1000000.0, solver='lbfgs',
                          max_iter=5000, random_state=seed)
            cal.fit(z, y)
            # cross fit for threshold
            for train_idx, valid_idx in StratifiedKFold(5, shuffle=True, random_state=7).split(z, y):
                cal1 = LogisticRegression(C=1000000.0, solver='lbfgs',
                              max_iter=5000, random_state=seed)
                cal1.fit(z[train_idx], y[train_idx])
                oof_cal[valid_idx] = cal1.predict_proba(z[valid_idx])[:, 1]
        fpr, tpr, cut = roc_curve(y, oof_cal, drop_intermediate=False)
        keep = np.isfinite(cut) & (cut <= 1)
        cut = float(cut[keep][np.argmax((tpr - fpr)[keep])])
        
        models, p_train, p_valid = [], np.zeros(len(y)), np.zeros(len(y_valid))
        for i in range(10):
            fit = model(name, p)
            fit.fit(x[i], y)
            models.append(fit)
            p_train += raw(fit, x[i], name) / 10
            p_valid += raw(fit, x_valid[i], name) / 10
        p_train, p_valid = apply_platt(cal, p_train, name), apply_platt(cal, p_valid, name)
        joblib.dump(dict(models=models, platt=cal, threshold=cut, cols=cols,
                         params=p, seed=seed,
                         platt_input=(None if cal is None else 'mean_decision' if name == 'SVM' else 'logit_mean_probability')),
                    f'model/{name}.joblib')
        out[name], out1[name], out2[name] = p_train, p_valid, oof_cal
        out2[name + '_raw'] = oof
        for group, y0, p0 in [('training', y, p_train),
                              ('temporal_validation', y_valid, p_valid)]:
            res.append(dict(model=name, cohort=group, **metrics(y0, p0, cut)))

        pd.DataFrame(res).to_csv("result/metrics.csv",index=False)
        out.to_csv('result/train.csv', index=False)
        out1.to_csv('result/validation.csv', index=False)
        out2.to_csv('result/oof.csv', index=False)
        with open('result/params.json', 'w',encoding='utf-8') as f:
            json.dump(params, f, indent=2)
if __name__ == '__main__':
    main()
