"""Recompute test metrics, patient-level bootstrap CIs, and ensemble comparisons
from saved probabilities. Usage: python analysis_ensembles.py DIR
DIR must contain probs_test_M{1,2,3,4}.npy, probs_test_M6_noSMOTE.npy, probs_test_M6.npy
(the SMOTE version), y_test.npy, test_idx.npy, patient_ids_12lead.npy.
Ensembles: M5 = mean(M1,M3,M4) (validation-selected), M5A = mean(M1..M4), M5T = mean(M2,M3,M4)
(test-selected, exploratory). Note: 'M5' in older summary/pairwise CSVs is M5A."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
D = sys.argv[1].rstrip('/') + '/'
K, B, SEED = 5, 10000, 42
y = np.load(D + 'y_test.npy'); pid = np.load(D + 'patient_ids_12lead.npy')[np.load(D + 'test_idx.npy')]
L = lambda n: np.load(D + n).astype(np.float64)
P = {k: L(f'probs_test_{k}.npy') for k in ['M1', 'M2', 'M3', 'M4']}
P['M6'] = L('probs_test_M6_noSMOTE.npy'); P['M6S'] = L('probs_test_M6.npy')
P['M5'] = (P['M1'] + P['M3'] + P['M4']) / 3
P['M5A'] = (P['M1'] + P['M2'] + P['M3'] + P['M4']) / 4
P['M5T'] = (P['M2'] + P['M3'] + P['M4']) / 3
Y = np.eye(K)[y]
def cm(p): m = np.zeros((K, K), int); np.add.at(m, (y, p.argmax(1)), 1); return m
def f1s(m):
    tp = np.diag(m).astype(float); pr = m.sum(0); rc = m.sum(1)
    prec = np.divide(tp, pr, out=np.zeros(K), where=pr > 0); rec = np.divide(tp, rc, out=np.zeros(K), where=rc > 0)
    return np.divide(2*prec*rec, prec+rec, out=np.zeros(K), where=(prec+rec) > 0), prec, rec
def ece(p, n=15):
    c = p.max(1); ok = p.argmax(1) == y; b = np.minimum((c*n).astype(int), n-1)
    return sum((b == i).sum()/len(y)*abs(ok[b == i].mean()-c[b == i].mean()) for i in range(n) if (b == i).any())
def summary(p):
    m = cm(p); f, pr, rc = f1s(m)
    return dict(acc=np.trace(m)/len(y), macro_f1=f.mean(), prec=pr.mean(), rec=rc.mean(),
        auroc=np.mean([roc_auc_score(Y[:, k], p[:, k]) for k in range(K)]),
        auprc=np.mean([average_precision_score(Y[:, k], p[:, k]) for k in range(K)]),
        ece=ece(p), brier=((p-Y)**2).sum(1).mean(), nll=-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
up, inv = np.unique(pid, return_inverse=True); NP = len(up)
W = np.random.default_rng(SEED).multinomial(NP, np.ones(NP)/NP, size=B).astype(float)
def boot_f1(p):
    a = np.zeros((NP, K*K)); np.add.at(a, (inv, y*K+p.argmax(1)), 1)
    M = (W @ a).reshape(B, K, K); tp = np.einsum('bkk->bk', M); pr = M.sum(1); rc = M.sum(2)
    prec = np.divide(tp, pr, out=np.zeros_like(tp), where=pr > 0); rec = np.divide(tp, rc, out=np.zeros_like(tp), where=rc > 0)
    return np.divide(2*prec*rec, prec+rec, out=np.zeros_like(tp), where=(prec+rec) > 0).mean(1)
BF = {k: boot_f1(p) for k, p in P.items()}; S = {k: summary(p) for k, p in P.items()}
rows = [dict(model=k, **S[k], f1_lo=np.percentile(BF[k], 2.5), f1_hi=np.percentile(BF[k], 97.5)) for k in P]
pd.set_option('display.width', 220); print(pd.DataFrame(rows).set_index('model').round(4))
print('\nPaired macro-F1 differences (A-B): delta, 95% CI, 90% CI')
for a, b in [('M5','M1'),('M5','M2'),('M5','M3'),('M5','M4'),('M5','M6'),('M5','M5T'),('M5A','M6'),('M5T','M6'),('M2','M3'),('M2','M6'),('M3','M6')]:
    x = BF[a]-BF[b]; print(f"{a}-{b}: {S[a]['macro_f1']-S[b]['macro_f1']:+.3f}  95% [{np.percentile(x,2.5):+.3f},{np.percentile(x,97.5):+.3f}]  90% [{np.percentile(x,5):+.3f},{np.percentile(x,95):+.3f}]")
ab = y != 3
for k in ['M5', 'M5T', 'M6']: print(k, 'false-NORM', int(((P[k].argmax(1) == 3) & ab).sum()), 'of', int(ab.sum()))
