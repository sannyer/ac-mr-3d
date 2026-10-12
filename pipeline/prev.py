import sys,pickle,numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
C={'clav':(0.91,0.86,0.76),'acro':(0.8,0.73,0.57),'cora':(0.66,0.76,0.65),'hum':(0.73,0.66,0.79)}
fs=sys.argv[1].split(','); views=[(5,-90,'front'),(80,-90,'top'),(20,-140,'oblique')]
fig=plt.figure(figsize=(6*len(views),6*len(fs)),facecolor='k')
for r,f in enumerate(fs):
    M=pickle.load(open(f,'rb'))
    for c,(el,az,nm) in enumerate(views):
        ax=fig.add_subplot(len(fs),len(views),r*len(views)+c+1,projection='3d'); ax.set_facecolor('k')
        for b,(V,F) in M.items():
            P=V[:,[0,2,1]]
            ax.add_collection3d(Poly3DCollection(P[F],facecolors=[C[b]],edgecolor='none',shade=True,lightsource=matplotlib.colors.LightSource(azdeg=225,altdeg=45)))
        ax.set_xlim(-28,38); ax.set_ylim(-33,33); ax.set_zlim(-40,26); ax.view_init(el,az); ax.axis('off'); ax.set_title(f.split('/')[-1]+' '+nm,color='w')
plt.tight_layout(); plt.savefig(sys.argv[2],dpi=70,facecolor='k')
