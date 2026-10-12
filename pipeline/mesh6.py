"""Bone meshes from fused SDFs: upsample x2 (cubic), marching cubes at 0.25 mm, light Taubin smoothing."""
import sys, os, pickle, numpy as np
from srr_common import HALF, VOX, N, HERE
from skimage.measure import marching_cubes
from scipy import ndimage as ndi
from taubin import taubin
phis = pickle.load(open(HERE+'/'+sys.argv[1],'rb'))
UP = float(os.environ.get('UP','1.5'))
IT = int(sys.argv[2]) if len(sys.argv)>2 else 12
VAL = dict(clav=2,acro=3,cora=4,hum=5)
lab = np.zeros((N,N,N),np.uint8); best = np.full((N,N,N),np.inf,np.float32); meshes = {}
g = (np.arange(N)+0.5)*VOX-HALF
for b, phi in phis.items():
    phi = phi.copy()
    gs = float(os.environ.get('GS_'+b, os.environ.get('GS','0')))
    if gs>0: phi = ndi.gaussian_filter(phi, gs/VOX)
    if b=='clav': phi = np.maximum(phi,(g-30.0)[:,None,None])
    ins = phi<0; l,k = ndi.label(ins)
    if k>1:
        keep = l==(np.argmax(ndi.sum(ins,l,range(1,k+1)))+1); phi = np.where(ins&~keep,np.abs(phi)+0.1,phi)
    for a in range(3):
        sl=[slice(None)]*3; sl[a]=[0,-1]; phi[tuple(sl)] = np.maximum(phi[tuple(sl)],0.5)
    idx = np.argwhere(phi<1.5); lo = np.maximum(idx.min(0)-3,0); hi = np.minimum(idx.max(0)+4,N)
    sub = phi[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]
    cs = [np.arange(0,n-1+1e-6,1/UP) for n in sub.shape]
    up = ndi.map_coordinates(ndi.spline_filter(sub,3), np.meshgrid(*cs,indexing='ij'), order=3, prefilter=False)   # 0.25 mm
    V,F,_,_ = marching_cubes(up,0.0)
    V = (V/UP + lo + 0.5)*VOX - HALF
    V = taubin(V,F,it=IT)
    meshes[b] = (V.astype(np.float32),F.astype(np.int32))
    m = (phi<0)&(phi<best); lab[m]=VAL[b]; best[m]=phi[m]
    print(b,'verts',len(V),'vol ml',round(float((phi<0).sum()*VOX**3/1000),2),'extent',V.min(0).round(1),V.max(0).round(1))
tag = os.environ.get('TAG','a')
pickle.dump(meshes,open(HERE+f'/meshes6_{tag}.pkl','wb')); np.save(HERE+f'/bonelab6_{tag}.npy',lab)
