"""Contour distance between a 3D bone label volume projected into native slices and the 2D contours on those slices."""
import sys, os, pickle, numpy as np
os.environ.setdefault('LABF','lab_v3.npy')
import seg6
from seg6 import *
D=pickle.load(open(HERE+'/'+os.environ.get('SEGF','seg6.pkl'),'rb')); seg,info=D['seg'],D['info']
sers=sys.argv[2].split(',')
for lf in sys.argv[1].split(','):
    seg6.LAB=np.load(HERE+'/'+lf); res={}
    for ser in sers:
        st=S[ser]; sp=st['sp']
        for k,bm in seg.get(ser,{}).items():
            L=proj_lab(st,k)
            for b,m in bm.items():
                if info[ser][k][b]['iou']<0.55: continue
                p=L==VAL[b]
                if not p.any(): continue
                e1=m&~ndi.binary_erosion(m); e2=p&~ndi.binary_erosion(p)
                d1=ndi.distance_transform_edt(~e2)[e1]*sp; d2=ndi.distance_transform_edt(~e1)[e2]*sp
                res.setdefault(b,[]).append(np.r_[d1,d2])
    print(lf, {b:(round(float(np.median(np.concatenate(v))),2), round(float(np.mean(np.concatenate(v))),2)) for b,v in res.items()}, 'all median', round(float(np.median(np.concatenate([np.concatenate(v) for v in res.values()]))),2))
