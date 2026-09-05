"""Offline, label-oracle repairability audit for the fixed strong-client pool."""
import argparse, csv, json, os, pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / 'data' / 'cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json'
OUT = ROOT / 'results' / 'strong_pool_repairability'
CIFAR = Path(os.environ.get('SEMCONSFL_CIFAR_ROOT', r'C:\Users\admin\Desktop\paper_ori\all_data\data'))
LAMBDAS = [0., 1e-4, 1e-3, 1e-2, 1e-1, 1., 10.]

def jsd(p,q):
    m=(p+q)/2; kl=lambda a,b: np.sum(a[a>0]*np.log2(a[a>0]/b[a>0]))
    return float((kl(p,m)+kl(q,m))/2)
def metrics(p,g):
    return {'tvd':float(np.abs(p-g).sum()/2),'jsd':jsd(p,g),'l2_squared':float(np.sum((p-g)**2))}
def summary(v):
    return {k:float(np.percentile(v,q)) for k,q in [('mean',50),('median',50),('p5',5),('p25',25),('p75',75),('p95',95)]} | {'mean':float(np.mean(v))}
def project_simplex(v):
    u=np.sort(v)[::-1]; cssv=np.cumsum(u)-1; rho=np.nonzero(u > cssv/np.arange(1,len(v)+1))[0][-1]
    theta=cssv[rho]/(rho+1); return np.maximum(v-theta,0)
def solve_qp(ps,g,alpha0,lam):
    # Projected-gradient solve of the convex simplex QP; no RNG or labels beyond this oracle objective.
    gram=np.sum(ps[:,None,:]*ps[None,:,:],axis=2)
    lipschitz=2*np.sum(np.abs(gram),axis=1).max()+2*lam; step=1/lipschitz
    x=alpha0.copy()
    for iteration in range(200000):
        grad=2*(np.sum(gram*x[None,:],axis=1)-np.sum(ps*g[None,:],axis=1))+2*lam*(x-alpha0); nxt=project_simplex(x-step*grad)
        if np.max(np.abs(nxt-x)) < 1e-13: return nxt, iteration+1, True
        x=nxt
    return x, iteration+1, False
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--simulations',type=int,default=5000); ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--output-dir',type=Path,default=OUT); a=ap.parse_args()
    payload=json.loads(SPLIT.read_text()); users={int(k):v for k,v in payload['train_data'].items()}
    with (CIFAR/'cifar-100-python'/'train').open('rb') as f: labels=np.asarray(pickle.load(f,encoding='latin1')['fine_labels'])
    H=np.asarray([np.bincount(labels[np.asarray(users[i])],minlength=100) for i in range(100)],float); n=H.sum(1); P=H/n[:,None]; g=H.sum(0)/H.sum()
    ids=np.arange(70,100); hs=H[ids]; ps=P[ids]; ns=n[ids]; alpha0=ns/ns.sum(); basep=np.sum(alpha0[:,None]*ps,axis=0); base=metrics(basep,g)
    records=[]; rng=np.random.RandomState(a.seed)
    for lam in LAMBDAS:
        alpha, iterations, converged=solve_qp(ps,g,alpha0,lam)
        if not converged: raise RuntimeError('projected-gradient QP failed to converge for lambda {}'.format(lam))
        alpha=np.maximum(alpha,0); alpha/=alpha.sum(); pi=(alpha/ns); pi/=pi.sum(); m=metrics(np.sum(alpha[:,None]*ps,axis=0),g)
        ess_a=1/np.sum(alpha**2); ess_p=1/np.sum(pi**2); order=np.sort(pi)[::-1]
        rec={'lambda':lam,'converged':converged,'iterations':iterations,'objective':float(m['l2_squared']+lam*np.sum((alpha-alpha0)**2)),'alpha':alpha.tolist(),'pi':pi.tolist(),**m,
             'ess_alpha':float(ess_a),'ess_pi':float(ess_p),'max_pi':float(order[0]),'top3_pi_mass':float(order[:3].sum()),'top5_pi_mass':float(order[:5].sum()),'entropy_pi_bits':float(-np.sum(pi[pi>0]*np.log2(pi[pi>0]))),
             'tvd_reduction_percent':float(100*(base['tvd']-m['tvd'])/base['tvd']),'jsd_reduction_percent':float(100*(base['jsd']-m['jsd'])/base['jsd'])}
        tv=[];jd=[]
        for _ in range(a.simulations):
            cnt=rng.multinomial(722,pi); q=np.sum(cnt[:,None]*hs,axis=0); q/=q.sum(); z=metrics(q,g);tv.append(z['tvd']);jd.append(z['jsd'])
        rec['finite_sampling']={'occurrences':722,'simulations':a.simulations,'tvd':summary(np.asarray(tv)),'jsd':summary(np.asarray(jd))}
        records.append(rec)
    # Uniform strong client sampling is sample-weighted in expectation because aggregation weighs n_i.
    piu=np.repeat(1/30,30); tv=[];jd=[]
    for _ in range(a.simulations):
        cnt=rng.multinomial(722,piu); q=np.sum(cnt[:,None]*hs,axis=0); q/=q.sum(); z=metrics(q,g);tv.append(z['tvd']);jd.append(z['jsd'])
    uniform={'alpha':alpha0.tolist(),'pi':piu.tolist(),**base,'ess_alpha':float(1/np.sum(alpha0**2)),'ess_pi':30.,'max_pi':1/30.,'top3_pi_mass':.1,'top5_pi_mass':1/6,'finite_sampling':{'occurrences':722,'simulations':a.simulations,'tvd':summary(np.asarray(tv)),'jsd':summary(np.asarray(jd))}}
    strict=[r for r in records if r['ess_pi']>=15 and r['max_pi']<=.15]
    relaxed=[r for r in records if r['ess_pi']>=10 and r['max_pi']<=.20]
    best=min(strict or relaxed or records,key=lambda r:r['l2_squared']); tier='strict' if strict else 'relaxed' if relaxed else 'concentrated'
    stable=(best['finite_sampling']['tvd']['mean']<uniform['finite_sampling']['tvd']['mean'] and best['finite_sampling']['jsd']['mean']<uniform['finite_sampling']['jsd']['mean'])
    if best['ess_pi']>=15 and best['max_pi']<=.15 and best['tvd_reduction_percent']>=30 and best['jsd_reduction_percent']>=30 and stable: cls='STRONGLY_REPAIRABLE'
    elif best['ess_pi']>=10 and best['max_pi']<=.20 and min(best['tvd_reduction_percent'],best['jsd_reduction_percent'])>=15 and stable: cls='PARTIALLY_REPAIRABLE'
    else: cls='POORLY_REPAIRABLE'
    out={'input':{'split':str(SPLIT),'label_file':str(CIFAR/'cifar-100-python'/'train'),'simulations':a.simulations},'uniform_strong_baseline':uniform,'candidates':records,'best_non_concentrated_candidate':best,'candidate_constraint_tier':tier,'classification':cls,'classification_note':'Label-oracle mechanism audit only; no candidate defines a training policy.'}
    a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/'repairability.json').write_text(json.dumps(out,indent=2))
    fields=['lambda','tvd','jsd','l2_squared','ess_alpha','ess_pi','max_pi','top3_pi_mass','top5_pi_mass','tvd_reduction_percent','jsd_reduction_percent']
    with (a.output_dir/'pareto.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in records])
    lines=['# Strong-pool repairability','',f"Classification: **{cls}**",'',f"Uniform pooled baseline: TVD {base['tvd']:.6f}; JSD {base['jsd']:.6f} bits.",'', '| lambda | TVD | JSD | ESS alpha | ESS pi | max pi | top-3 | top-5 | TVD reduction | JSD reduction |','| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in records: lines.append('| {lambda:g} | {tvd:.6f} | {jsd:.6f} | {ess_alpha:.2f} | {ess_pi:.2f} | {max_pi:.4f} | {top3_pi_mass:.4f} | {top5_pi_mass:.4f} | {tvd_reduction_percent:.2f}% | {jsd_reduction_percent:.2f}% |'.format(**r))
    lines += ['', '## Selected diversity-constrained candidate','',f"Constraint tier: {tier}; lambda={best['lambda']:g}; TVD/JSD={best['tvd']:.6f}/{best['jsd']:.6f}; ESS alpha/pi={best['ess_alpha']:.2f}/{best['ess_pi']:.2f}; max pi={best['max_pi']:.4f}.",'',f"Finite 722-occurrence TVD mean: uniform {uniform['finite_sampling']['tvd']['mean']:.6f}, candidate {best['finite_sampling']['tvd']['mean']:.6f}; JSD mean: uniform {uniform['finite_sampling']['jsd']['mean']:.6f}, candidate {best['finite_sampling']['jsd']['mean']:.6f}.", '', 'This is a label-oracle upper-bound analysis; labels are not used to define any training mechanism.']
    (a.output_dir/'repairability.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'classification':cls,'best_lambda':best['lambda'],'tvd':best['tvd'],'jsd':best['jsd'],'ess_pi':best['ess_pi'],'max_pi':best['max_pi']}))
if __name__=='__main__': main()
