from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss

def predict(df, fit):
    x = df[fit['cols']].to_numpy(dtype=float)
    p=(np.zeros(len(df)))
    for model in fit['models']:
        p += model.predict_proba(x)[:, 1] / len(fit['models'])
    z = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return fit['platt'].predict_proba(z)[:, 1]
def score(y, p, cut):
    tn, fp, fn, tp = confusion_matrix(y, p >= cut, labels=[0, 1]).ravel()
    div = lambda a, b: a / b if b else np.nan
    rng = np.random.default_rng(1234)
    neg, pos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    auc = []
    for i in range(2000):
        idx = np.r_[rng.choice(neg, len(neg), replace=True),
                    rng.choice(pos, len(pos), replace=True)]
        auc.append(roc_auc_score(y[idx], p[idx]))
    lo,hi = np.quantile(auc,
                  [0.025,0.975])
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    cal = sm.GLM(y, np.ones((len(y), 1)),
                  family=sm.families.Binomial(), offset=z).fit()
    ci = cal.conf_int(alpha=0.05)
    cal1 = sm.GLM(y, sm.add_constant(z),
                  family=sm.families.Binomial()).fit()
    ci1 = cal1.conf_int(alpha=0.05)
    return dict(n=len(y), positive=int(y.sum()), negative=int((y == 0).sum()),
                prevalence=float(y.mean()), AUC=roc_auc_score(y, p), lower=lo, upper=hi,
                sensitivity=div(tp, tp + fn), specificity=div(tn, tn + fp),
                accuracy=div(tp + tn, len(y)), F1=div(2 * tp, 2 * tp + fp + fn),
                PPV=div(tp, tp + fp), NPV=div(tn, tn + fn),
                Brier=brier_score_loss(y, p), threshold=float(cut),
                TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
                prioritized=int(tp + fp), prioritized_fraction=div(tp + fp, len(y)),
                intercept=cal.params[0], intercept_lower=ci[0, 0], intercept_upper=ci[0, 1],
                slope=cal1.params[1], slope_lower=ci1[1, 0], slope_upper=ci1[1, 1])

def main():
    fit=joblib.load("model/RF.joblib")
    cut = fit['threshold']
    df = pd.read_csv('ext/a.csv', dtype={'SampleID': str}).set_index('SampleID')
    cc = pd.read_csv('ext/cc.csv', dtype={'SampleID': str}).set_index('SampleID')
    Path('ext').mkdir(exist_ok=True)
    # complete 
    p = predict(cc, fit)
    y = cc.IDH.to_numpy(dtype=int)
    res = [dict(cohort='complete_case', **score(y, p, cut))]
    q = pd.DataFrame({'y': y, 'p': p})
    q['group'] = pd.qcut(q.p, 5, duplicates='drop')
    q = q.groupby('group', observed=True).agg(
        predicted=('p', 'mean'), observed=('y', 'mean'),
        n=('y', 'size'), events=('y', 'sum')).reset_index(drop=True)
    q.insert(0, 'quintile', np.arange(1, len(q) + 1))
    q.to_csv('ext/q.csv',index=False)
    # DCA
    cuts = np.arange(1, 100) / 100
    keep = p[:, None] >= cuts[None, :]
    tp = (keep & (y[:, None] == 1)).sum(axis=0)
    fp = (keep & (y[:, None] == 0)).sum(axis=0)
    nb = tp / len(y) - fp / len(y) * cuts / (1 - cuts)
    all_nb = y.mean() - (1 - y.mean()) * cuts / (1 - cuts)
    pd.DataFrame({'threshold': cuts, 'RF': nb, 'all': all_nb,
                  'none': np.zeros(len(cuts))}).to_csv('ext/d.csv', index=False)

    prob = np.zeros(len(df))
    for i in range(1, 11):
        dat = pd.read_csv(f"ext/x{i}.csv",
                  dtype={'SampleID': str}).set_index('SampleID')
        dat = dat.loc[df.index]
        x = dat[fit['cols']].to_numpy(dtype=float)
        p = np.zeros(len(dat))
        for model in fit['models']:
            p += model.predict_proba(x)[:, 1] / len(fit['models'])
        prob += p / 10
    z = logit(np.clip(prob, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    prob = fit['platt'].predict_proba(z)[:, 1]
    y = df.IDH.to_numpy(dtype=int)
    for k in [1, 2, 3]:
        keep = df['miss'].to_numpy() <= k
        res.append(dict(cohort=f'missing_le{k}', **score(y[keep], prob[keep], cut)))
    pd.DataFrame(res).to_csv('ext/m.csv', index=False)

if __name__ == '__main__':
    main()
