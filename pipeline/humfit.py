"""Fit a sphere to the humeral-head cortex arcs (upper parts of the 2D contours from all six series) and build a full-head prior."""
import pickle, numpy as np
from seg6 import *
D=pickle.load(open(HERE+'/seg6.pkl','rb'))['seg']; P=[]
for ser,kk in D.items():
    for k,bm in kk.items():
        if 'hum' not in bm: continue
        m=bm['hum']; e=m&~ndi.binary_erosion(m); W=slice_world(S[ser],k)[e]
        P.append(W[W[:,1]>W[:,1].min()+4])
P=np.concatenate(P); print(len(P))
def fit(P):
    A=np.c_[2*P,np.ones(len(P))]; b=(P**2).sum(1); x=np.linalg.lstsq(A,b,rcond=None)[0]
    c=x[:3]; r=np.sqrt(x[3]+c@c); return c,r
keep=np.ones(len(P),bool)
for it in range(6):
    c,r=fit(P[keep]); d=np.abs(np.linalg.norm(P-c,axis=1)-r); keep=d<max(1.0,2.5*np.median(d[keep]))
print('centre',c.round(1),'r',round(r,1),'resid median',round(float(np.median(d[keep])),2),'kept',keep.mean().round(2))
g=(np.arange(N)+0.5)*VOX-HALF; X,Y,Z=np.meshgrid(g,g,g,indexing='ij')
ball=((X-c[0])**2+(Y-c[1])**2+(Z-c[2])**2<r*r)&(Y>-43)
L=np.load(HERE+'/lab_v3.npy'); L2=L.copy(); L2[L2==5]=0; L2[ball&(L2==0)]=5
np.save(HERE+'/lab_humball.npy',L2); np.save(HERE+'/humsphere.npy',np.r_[c,r])
