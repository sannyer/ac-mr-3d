import sys,pickle,numpy as np
from PIL import Image, ImageDraw
from seg6 import *
d=pickle.load(open(HERE+'/'+os.environ.get('SEGF','seg6.pkl'),'rb'))['seg']
COL={'clav':(80,170,255),'acro':(255,200,60),'cora':(120,230,120),'hum':(230,120,255)}
ser=sys.argv[1]; ks=[int(x) for x in sys.argv[2].split(',')]; out=sys.argv[3]
st=S[ser]; tiles=[]
for k in ks:
    L=proj_lab(st,k); a=st['img'][k]
    # crop around bones
    ys,xs=np.nonzero(L>=2); cy,cx=int(ys.mean()),int(xs.mean()); h=int(32/st['sp'])
    j0,i0=max(0,cy-h),max(0,cx-h); j1,i1=min(a.shape[0],cy+h),min(a.shape[1],cx+h)
    g=np.clip(a/np.percentile(a,99.5),0,1)*255; rgb=np.stack([g]*3,-1)[j0:j1,i0:i1].copy()
    for b,v in VAL.items():
        m=(L==v)[j0:j1,i0:i1]; e=m&~ndi.binary_erosion(m); rgb[e]=np.array(COL[b])*0.45
    for b,m in d.get(ser,{}).get(k,{}).items():
        m=m[j0:j1,i0:i1]; e=m&~ndi.binary_erosion(m); rgb[e]=COL[b]
    im=Image.fromarray(rgb.astype(np.uint8)).resize((520,520),Image.NEAREST)
    ImageDraw.Draw(im).text((4,4),f'{ser}#{k}',fill=(255,255,0)); tiles.append(im)
c=3; r=(len(tiles)+c-1)//c; sh=Image.new('RGB',(c*520,r*520))
for i,t in enumerate(tiles): sh.paste(t,((i%c)*520,(i//c)*520))
sh.save(out)
