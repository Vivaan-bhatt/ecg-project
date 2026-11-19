import numpy as np, pandas as pd, ast, sys
from sklearn.metrics import f1_score
R='/home/claude/vivaan-bhatt/ecg-project/'; A='/mnt/user-data/attachments/cdcb89f2-fd6f-5b8f-b451-ef0ab2d9fa59/'
db=pd.read_csv(A+'bf6d24bb-ptbxl_database.csv').set_index('ecg_id')
sc=pd.read_csv(A+'edac10fb-scp_statements.csv',index_col=0)
diag=sc[sc.diagnostic==1.0]['diagnostic_class'].to_dict()
cls=['CD','HYP','MI','NORM','STTC']; ci={c:i for i,c in enumerate(cls)}
rid=np.load(R+'record_ids_raw_12lead.npy'); yraw=np.load(R+'y_raw_12lead.npy'); te=np.load(R+'test_idx.npy'); pid=np.load(R+'patient_ids_12lead.npy',allow_pickle=True)
first=[];last=[];sets=[]
for r in rid:
    d=ast.literal_eval(db.loc[r,'scp_codes'])
    items=[(k,v) for k,v in d.items() if k in diag and v>0]   # dict order = listed order
    if not items: first.append(-1);last.append(-1);sets.append(set());continue
    mx=max(v for k,v in items); top=[k for k,v in items if v==mx]
    first.append(ci[diag[top[0]]]);last.append(ci[diag[top[-1]]]);sets.append({ci[diag[k]] for k,v in items})
first=np.array(first);last=np.array(last);nset=np.array([len(s) for s in sets])
print("records w/o diag code:",(first<0).sum(),"| first==y_raw:",(first==yraw).mean(),"mismatch n:",(first!=yraw).sum())
print("multi-superclass share (all):",(nset>1).mean(),"| tie-dependent (first!=last):",(first!=last).mean())
print("HYP total derived now:",(first==1).sum(),"y_raw:",(yraw==1).sum())
y=np.load(R+'y_test.npy'); assert (yraw[te]==y).all()
P={k:np.load(R+f'probs_test_{k}.npy') for k in ['M1','M2','M3','M4','M6_noSMOTE']}
P['M5']=(P['M1']+P['M3']+P['M4'])/3; P['M5A']=(P['M1']+P['M2']+P['M3']+P['M4'])/4; P['M5T']=(P['M2']+P['M3']+P['M4'])/3
ns=nset[te]; sets_te=[sets[i] for i in te]; last_te=last[te]; first_te=first[te]
print("test: single-superclass n=",(ns==1).sum(),"multi n=",(ns>1).sum(),"tie-dependent n=",(first_te!=last_te).sum())
def mf(yt,p,m=None):
    if m is not None: yt=yt[m];p=p[m]
    return f1_score(yt,p,average='macro',labels=range(5))
print("model | all | single-only | multi-only | last-listed labels | set-match acc | acc(first)")
rows={}
for k in ['M1','M2','M3','M4','M5','M5A','M5T','M6_noSMOTE']:
    pr=P[k].argmax(1)
    sm=np.mean([pr[i] in sets_te[i] for i in range(len(pr))])
    rows[k]=(mf(y,pr),mf(y,pr,ns==1),mf(y,pr,ns>1),mf(last_te,pr,last_te>=0),sm,(pr==y).mean())
    print(k,*[round(v,3) for v in rows[k]])
# cluster bootstrap for single-only M5-M6, last-listed M5-M6
rng=np.random.default_rng(42); pt=pid[te]; u,inv=np.unique(pt,return_inverse=True)
idx_by=[np.where(inv==j)[0] for j in range(len(u))]
def boot(ya,pa,pb,mask,B=2000):
    d=[]
    for _ in range(B):
        s=rng.integers(0,len(u),len(u)); ii=np.concatenate([idx_by[j] for j in s]); ii=ii[mask[ii]]
        d.append(f1_score(ya[ii],pa[ii],average='macro',labels=range(5))-f1_score(ya[ii],pb[ii],average='macro',labels=range(5)))
    return np.percentile(d,[2.5,97.5])
p5=P['M5'].argmax(1);p6=P['M6_noSMOTE'].argmax(1)
print("M5-M6 single-only:",round(rows['M5'][1]-rows['M6_noSMOTE'][1],3),boot(y,p5,p6,ns==1).round(3))
print("M5-M6 last-listed:",round(rows['M5'][3]-rows['M6_noSMOTE'][3],3),boot(last_te,p5,p6,last_te>=0).round(3))
