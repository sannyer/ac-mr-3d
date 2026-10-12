"""Fuse 2D contours from all six series into one 3D signed-distance field per bone (see v2 sdf_fuse.py).
Adds 'outside' evidence on slices just beyond each bone's extent. usage: fuse6.py LAM ITERS"""
import sys, os, pickle, time, numpy as np
import srr2
from srr2 import build_op, fwd, adj, LinearOperator, cg
from srr_common import *
srr2.NS = 3
T = 5.0
LAM = float(sys.argv[1]); ITERS = int(sys.argv[2]); MU = 2e-4
BONES = ['clav','acro','cora','hum']
S = pickle.load(open(HERE+'/stacks03.pkl','rb'))
D = pickle.load(open(HERE+'/'+os.environ.get('SEGF','seg6.pkl'),'rb')); seg, info = D['seg'], D['info']
MINIOU = float(os.environ.get('MINIOU','0.55'))
def lap(v):
    out = -6*v
    for a in range(3): out = out + np.roll(v,1,a) + np.roll(v,-1,a)
    return out
def sdf2d(m, sp):
    if not m.any(): return np.full(m.shape, T, np.float32)
    d = ndi.distance_transform_edt(~m)*sp - ndi.distance_transform_edt(m)*sp
    return np.clip(d,-T,T).astype(np.float32)
GP = grid_points(); t0 = time.time(); phis = {}
only = sys.argv[3].split(',') if len(sys.argv)>3 else BONES
for b in only:
    ops, inits = [], []
    for ser, st in S.items():
        if ser in os.environ.get('HOLD','').split(','): continue
        ks = [k for k in seg.get(ser,{}) if b in seg[ser][k] and info[ser][k][b]['iou']>=MINIOU]
        if not ks: continue
        allk = [k for k in seg.get(ser,{}) if b in seg[ser][k]]
        neigh = {min(allk)-1, max(allk)+1} - set(allk)
        img = np.full(st['img'].shape, np.nan, np.float32)
        SM = float(os.environ.get('SMOOTH','0'))
        for k in ks:
            q = sdf2d(seg[ser][k][b], st['sp'])
            img[k] = ndi.gaussian_filter(q, SM/st['sp']) if SM>0 else q
        for k in neigh:
            if 0 <= k < len(img): img[k] = T
        st2 = dict(st); st2['img'] = img
        kk, jj, ii = stack_coords(st, GP)
        inits.append(ndi.map_coordinates(img,[kk,jj,ii],order=1,mode='constant',cval=np.nan))
        op = build_op(st2); ok = ~np.isnan(op['y'])
        ops.append({q:(v[ok] if isinstance(v,np.ndarray) else v) for q,v in op.items()})
        print(b, ser, 'slices', sorted(ks), 'outside', sorted(neigh), flush=True)
    with np.errstate(all='ignore'): x0 = np.nanmean(np.stack(inits),0)
    x0 = np.where(np.isnan(x0), T, x0).astype(np.float32)
    x0 = ndi.gaussian_filter(x0.reshape(N,N,N),1).ravel()
    bvec = sum(adj(o,o['y']) for o in ops) + MU*T
    def mv(v):
        v = v.astype(np.float32); V = v.reshape(N,N,N)
        return sum(adj(o,fwd(o,v)) for o in ops) + LAM*lap(lap(V)).ravel() + MU*v
    x, _ = cg(LinearOperator((N**3,N**3),matvec=mv,dtype=np.float32), bvec, x0=x0, maxiter=ITERS)
    r = {o['ser']: round(float(np.sqrt(((fwd(o,x)-o['y'])**2).mean())),3) for o in ops}
    phis[b] = x.reshape(N,N,N).astype(np.float32)
    print(b,'rms misfit mm',r,'vol ml',round(float((phis[b]<0).sum()*VOX**3/1000),2),round(time.time()-t0),'s',flush=True)
pickle.dump(phis, open(HERE+'/phi6_%s.pkl'%os.environ.get('TAG','a'),'wb'))
