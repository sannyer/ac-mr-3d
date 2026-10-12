import json, base64, numpy as np, pickle
from srr_common import *
d=json.load(open('live_data.json'))
dims=d['dims']; o=np.array(d['origin'])
lo=np.round((o+HALF)/VOX-0.5).astype(int)
def vol(s): return np.frombuffer(base64.b64decode(s),np.uint8).reshape(dims[2],dims[1],dims[0]).transpose(2,1,0)
def rle(s):
    a=np.frombuffer(base64.b64decode(s),np.uint32).reshape(-1,2)
    return np.repeat(a[:,0],a[:,1]).astype(np.uint8).reshape(dims[2],dims[1],dims[0]).transpose(2,1,0)
full=lambda a: (lambda F:(F.__setitem__(tuple(slice(lo[i],lo[i]+dims[i]) for i in range(3)),a),F)[1])(np.zeros((N,N,N),a.dtype))
pd=full(vol(d['pd'])).astype(np.float32)/255; fat=full(vol(d['fat'])).astype(np.float32)/255; lab=full(rle(d['labrle']))
np.save(HERE+'/pd8.npy',pd); np.save(HERE+'/fat8.npy',fat); np.save(HERE+'/lab_v3.npy',lab)
print(lo, np.unique(lab,return_counts=True))
meshes={}
for b,m in d['meshes'].items():
    V=np.frombuffer(base64.b64decode(m['v']),np.float32).reshape(-1,3); F=np.frombuffer(base64.b64decode(m['f']),np.uint32).reshape(-1,3)
    meshes[b]=(V,F); print(b,V.min(0),V.max(0))
pickle.dump(meshes,open(HERE+'/meshes_v3.pkl','wb'))
S={s:load_stack(s,0.3) for s in ['301','401','501','601','701','801']}
for s,st in S.items(): print(s,st['img'].shape,st['sp'],st['gap'],st['th'])
pickle.dump(S,open(HERE+'/stacks03.pkl','wb'))
