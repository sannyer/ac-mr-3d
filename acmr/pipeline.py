"""End-to-end generator: DICOM -> SRR volumes -> 2D bone contours -> fused SDFs -> meshes -> measurements -> viewer data.

Every stage caches its result in the work directory, so a later stage can be re-run alone (--from STAGE)."""
import json, os, pickle, time
import numpy as np
from . import backend, dicomio, srr, seg2d, fuse, humerus, mesh, measure, export
from .geometry import SERIES

STAGES = ['srr', 'contours', 'humerus', 'mesh', 'export']
OTHERS = ['clav', 'acro', 'cora']


class Ctx:
    def __init__(self, dicom, work, out, log=print):
        self.dicom, self.work, self.out, self.log = dicom, work, out, log
        os.makedirs(work, exist_ok=True)
        self.B = backend.get()
        self._stacks = None

    @property
    def stacks(self):
        if self._stacks is None:
            self._stacks = dicomio.load_all(self.dicom)
        return self._stacks

    def save(self, name, obj):
        with open(os.path.join(self.work, name + '.pkl'), 'wb') as f:
            pickle.dump(obj, f, protocol=4)

    def load(self, name):
        with open(os.path.join(self.work, name + '.pkl'), 'rb') as f:
            return pickle.load(f)


def stage_srr(c):
    S = c.stacks
    c.log('PD (301, 501, 701: three directions)')
    pd = srr.reconstruct([S[s] for s in ('301', '501', '701')], B=c.B, log=c.log)['volume']
    c.log('fat-bright T2 (601) and T1 (801), each guided by the PD edges')
    norm = lambda v: np.clip(v / np.percentile(v[v > 0.02], 99.5), 0, 1).astype(np.float32)
    pd = norm(pd)
    fat = srr.reconstruct_guided_pair([S['601'], S['801']], pd, B=c.B, log=c.log)
    c.save('srr', dict(pd=pd, fat=norm(fat)))


def _fuse_all(c, seg, info, bones, **kw):
    return {b: fuse.fuse_bone(c.stacks, seg, info, b, B=c.B, log=c.log, **kw)[0] for b in bones}


def stage_contours(c):
    S = c.stacks
    c.log('initial contours from the hand-read seeds (601, 801)')
    seg, info = seg2d.initial(S)
    phis = _fuse_all(c, seg, info, seg2d.BONES, lam=0.4)
    for rnd, kw in ((1, dict(lam=0.4)), (2, dict(lam=1.0, smooth=1.2))):
        lab = fuse.labels_from_phis(phis)
        c.log(f'refinement {rnd}: contours on every slice of all six series')
        seg, info = seg2d.refine(S, lab)
        phis = _fuse_all(c, seg, info, seg2d.BONES, **kw)
    c.save('contours', dict(seg=seg, info=info, phis=phis))


def stage_humerus(c):
    S = c.stacks
    d = c.load('contours'); seg, info, phis = d['seg'], d['info'], d['phis']
    fc, fr, res = humerus.fit_sphere(S, seg)
    c.log(f'sphere fitted to the current contours: centre {np.round(fc, 1)} r {fr:.1f} mm (median residual {res:.2f} mm)')
    cen, r = humerus.SPHERE
    c.log(f'humeral head prior: centre {cen} r {r} mm')
    lab = humerus.ball_prior(fuse.labels_from_phis(phis), cen, r)
    sh, ih = seg2d.refine(S, lab, ['hum'], ero=3.0, dil=4.0)
    ph = fuse.fuse_bone(S, sh, ih, 'hum', lam=1.0, iters=120, smooth=1.0, B=c.B, log=c.log)[0]
    sh, ih = seg2d.refine(S, humerus.replace(lab, ph), ['hum'], ero=2.0, dil=5.0)
    ph = fuse.fuse_bone(S, sh, ih, 'hum', lam=1.5, iters=160, smooth=1.5, min_iou=0.8, B=c.B, log=c.log)[0]
    seg, info = seg2d.merge((seg, info), (sh, ih), ['hum'])
    phis['hum'] = ph
    c.save('bones', dict(seg=seg, info=info, phis=phis, sphere=dict(centre=cen.tolist(), r=r)))


def stage_mesh(c):
    phis = c.load('bones')['phis']
    meshes, final = {}, {}
    for b, phi in phis.items():
        V, F, p = mesh.bone_mesh(phi, b, gs=1.5 if b == 'hum' else 0.6)
        meshes[b] = (V, F); final[b] = p
        c.log(f'  {b}: {len(V)} vertices, {(p < 0).sum() * 0.125 / 1000:.2f} ml')
    meas = measure.measure(meshes)
    c.log('measurements (non-diagnostic): ' + json.dumps(meas))
    c.save('mesh', dict(meshes=meshes, phis=final, labels=fuse.labels_from_phis(final), meas=meas))


def stage_export(c):
    v = c.load('srr'); m = c.load('mesh'); bones = c.load('bones')
    lab = export.soft_zone(m['labels'], v['pd'], m['meas'])
    der = os.path.join(c.out, 'derived'); vw = os.path.join(c.out, 'viewer')
    os.makedirs(os.path.join(der, 'meshes'), exist_ok=True); os.makedirs(vw, exist_ok=True)
    np.savez_compressed(os.path.join(der, 'volumes.npz'), pd=v['pd'].astype(np.float16), fat=v['fat'].astype(np.float16))
    np.savez_compressed(os.path.join(der, 'bones.npz'), labels=lab,
                        **{'sdf_' + b: np.clip(p, -10, 10).astype(np.float16) for b, p in m['phis'].items()})
    cont = {}
    for ser, kk in bones['seg'].items():
        for k, bm in kk.items():
            for b, mk in bm.items():
                if mk.any():
                    cont[f'{ser}_{k:02d}_{b}'] = np.packbits(mk, axis=None)
                    cont[f'{ser}_{k:02d}_{b}_shape'] = np.array(mk.shape, np.int16)
    np.savez_compressed(os.path.join(der, 'contours.npz'), **cont)
    for b, (V, F) in m['meshes'].items():
        mesh.write_ply(os.path.join(der, 'meshes', b + '.ply'), V, F)
    with open(os.path.join(der, 'measurements.json'), 'w') as f:
        json.dump(dict(m['meas'], humeral_head_sphere=bones['sphere'], note='geometric, non-diagnostic'), f, indent=1)
    nat = {s: export.native_atlas(c.stacks[s]) for s in SERIES}
    export.write_viewer_js(os.path.join(vw, 'ac3d_data.js'), export.viewer_data(v['pd'], v['fat'], lab, m['meshes'], m['meas'], nat))
    c.log(f'wrote {der} and {vw}/ac3d_data.js')


def run(dicom, work, out, start=None, stop=None, log=print):
    c = Ctx(dicom, work, out, log)
    log(f'backend: {c.B.name}')
    i0 = STAGES.index(start) if start else 0; i1 = STAGES.index(stop) + 1 if stop else len(STAGES)
    for s in STAGES[i0:i1]:
        t = time.time(); log(f'== {s}')
        globals()['stage_' + s](c)
        log(f'== {s} done in {time.time() - t:.0f} s')
