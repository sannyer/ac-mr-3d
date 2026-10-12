"""Inject the six-series bone model into the live viewer data: new meshes, bone labels + soft-tissue zone, measurements.
PD/fat volumes, raw-surface mask and native slices are kept from the current viewer data (live_data.json)."""
import json, base64, pickle, numpy as np
from scipy import ndimage as ndi
from srr_common import HALF, VOX, N, HERE
import os; TAG=os.environ.get('TAG','b')
d = json.load(open(HERE+'/live_data.json'))
dims = d['dims']; lo = np.round((np.array(d['origin'])+HALF)/VOX-0.5).astype(int)
lab0 = np.load(HERE+'/bonelab6_'+TAG+'.npy'); meas = json.load(open(HERE+'/measure6.json'))
pd = np.load(HERE+'/pd8.npy')
M = {b: lab0==v for b,v in dict(clav=2,acro=3,cora=4,hum=5).items()}
Dd = {b: ndi.distance_transform_edt(~m)*VOX for b,m in M.items()}
bone = lab0>0; gAC, gCC = meas['ac_gap_min'], meas['cc_min']
zone = ((Dd['clav']<4)|(Dd['acro']<4)|(Dd['cora']<3)|((Dd['clav']+Dd['acro'])<gAC+12)|((Dd['clav']+Dd['cora'])<gCC+8)) & ~bone
zone &= ndi.binary_fill_holes(ndi.gaussian_filter(pd,2)>0.08)
lab = np.zeros((N,N,N),np.uint8); lab[zone]=1; lab[bone]=lab0[bone]
sl = tuple(slice(lo[i],lo[i]+dims[i]) for i in range(3))
assert lab.sum()==lab[sl].sum(), 'labels outside crop'
L = np.ascontiguousarray(lab[sl].transpose(2,1,0)).ravel()
b64 = lambda a: base64.b64encode(a.tobytes()).decode()
ch = np.flatnonzero(np.diff(L))+1; st = np.r_[0,ch]; ln = np.diff(np.r_[st,len(L)])
d['labrle'] = b64(np.stack([L[st].astype(np.uint32),ln.astype(np.uint32)],1).ravel())
d['meshes'] = {}
for b,(V,F) in pickle.load(open(HERE+'/meshes6_'+TAG+'.pkl','rb')).items():
    assert len(V)<65536
    d['meshes'][b] = {'v':b64(V.astype(np.float32)),'f':b64(F.astype(np.uint16)),'nv':len(V),'nf':len(F)}
d['meas'] = meas
open(HERE+'/data6.js','w').write('<script>window.AC3D='+json.dumps(d,separators=(',',':'))+';</script>\n')
print('ok', round(len(json.dumps(d))/1e6,2),'MB')
