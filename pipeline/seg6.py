"""2D bone contours on native slices of ALL six diagnostic series, seeded from the current 3D bone model projected
into each slice (eroded = inside marker, dilated complement = outside marker). Random walker at native resolution."""
import sys, os, pickle, numpy as np
from multiprocessing import Pool
from srr_common import *
from skimage.segmentation import random_walker
BONES=['clav','acro','cora','hum']; VAL=dict(clav=2,acro=3,cora=4,hum=5)
S=pickle.load(open(HERE+'/stacks03.pkl','rb'))
LAB=np.load(HERE+'/'+os.environ.get('LABF','lab_v3.npy'))
ERO=float(os.environ.get('ERO','2.0'))
DIL=float(os.environ.get('DIL','2.5'))

def slice_world(st,k):
    Z,R,C=st['img'].shape
    jj,ii=np.mgrid[:R,:C]
    P=st['pos'][k]+np.outer(ii.ravel(),st['r'])*st['sp']+np.outer(jj.ravel(),st['c'])*st['sp']
    Rm,t=rigid(st['ser']); return ((P-t)@Rm).reshape(R,C,3)

def proj_lab(st,k):
    P=slice_world(st,k); v=world_to_vox(P.reshape(-1,3))
    return ndi.map_coordinates(LAB,v.T,order=0,cval=0).reshape(P.shape[:2])

def disk(r):
    r=max(1,int(round(r))); y,x=np.mgrid[-r:r+1,-r:r+1]; return x*x+y*y<=r*r

def job(a):
    ser,k=a; st=S[ser]; sp=st['sp']
    L=proj_lab(st,k)
    if not (L>=2).any(): return ser,k,{},{}
    img=ndi.gaussian_filter(st['img'][k].astype(np.float32),0.5)
    out={}; info={}
    for b in os.environ.get('ONLY',','.join(BONES)).split(','):
        m0=L==VAL[b]
        if m0.sum()*sp*sp<6: continue
        ys,xs=np.nonzero(m0); pad=int(12/sp)
        j0,j1=max(0,ys.min()-pad),min(L.shape[0],ys.max()+pad); i0,i1=max(0,xs.min()-pad),min(L.shape[1],xs.max()+pad)
        a_=img[j0:j1,i0:i1]; mm=m0[j0:j1,i0:i1]; Lc=L[j0:j1,i0:i1]
        ins=ndi.binary_erosion(mm,disk(ERO/sp))
        if ins.sum()*sp*sp<1.5: ins=ndi.binary_erosion(mm,disk(1.0/sp))
        if ins.sum()*sp*sp<0.5: continue
        outm=~ndi.binary_dilation(mm,disk(DIL/sp)) | ndi.binary_erosion((Lc>=2)&(Lc!=VAL[b]),disk(0.8/sp))
        ref=np.percentile(a_[ndi.binary_dilation(mm,disk(DIL/sp))],99)
        an=np.clip(a_/max(ref,1e-3),0,1.2)
        mk=np.zeros(a_.shape,np.int32); mk[outm]=2; mk[ins]=1
        try: lab=random_walker(an,mk,beta=1500,mode='cg_j',tol=1e-3)
        except Exception as e: continue
        m=ndi.binary_fill_holes(lab==1); m=ndi.binary_opening(m,disk(0.6/sp))
        l,n=ndi.label(m)
        if n>1:
            ids=set(np.unique(l[ins]))-{0}
            m=np.isin(l,list(ids)) if ids else m
        iou=(m&mm).sum()/max(1,(m|mm).sum())
        full=np.zeros(L.shape,bool); full[j0:j1,i0:i1]=m
        out[b]=full; info[b]=dict(iou=float(iou),a0=float(mm.sum()*sp*sp),a=float(m.sum()*sp*sp))
    return ser,k,out,info

if __name__=='__main__':
    jobs=[]
    for ser,st in S.items():
        z=st['pos']@st['n']
        for k in range(len(z)):
            # slice must intersect the bone ROI
            jobs.append((ser,k))
    with Pool(4) as p: res=p.map(job,jobs,chunksize=2)
    seg={}; inf={}
    for ser,k,o,i in res:
        if o: seg.setdefault(ser,{})[k]=o; inf.setdefault(ser,{})[k]=i
    pickle.dump(dict(seg=seg,info=inf),open(HERE+'/'+os.environ.get('OUT','seg6.pkl'),'wb'))
    for ser in sorted(inf):
        for k in sorted(inf[ser]):
            print(ser,k,{b:(round(v['iou'],2),round(v['a0']),round(v['a'])) for b,v in inf[ser][k].items()})
