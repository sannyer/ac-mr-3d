# AC-joint 3D — v4 bones from all six series (2026-10-11)

Viewer republished in place: https://claude.ai/artifact/VUqnZeoWW3v5AEEnUMgbcE (template `tpl6.html`; data line inserted as line 159, `<script>window.AC3D=...</script>`).
The v3 scripts were not saved to the project; v3's volumes and bone labels were recovered from the live viewer data with `dec.py` (needs the artifact HTML read to `live.html`, its line 159 parsed to `live_data.json`). PD/fat volumes, raw-surface mask and native slice textures in v4 are v3's, unchanged.
Run with `AC3D_WORK=<dir>` holding `dcm/` (181 DICOMs) and the v2 helpers (`srr_common.py`, `srr2.py`).

1. `dec.py` -> pd8.npy, fat8.npy, lab_v3.npy, stacks03.pkl (all 6 stacks at native in-plane).
2. `seg6.py` — random-walker 2D bone contour on every native slice of 301/401/501/601/701/801 that the prior model touches (prior eroded 2 mm = inside, dilated 2.5 mm complement = outside). ~150 contours. QC: `qc6.py SER k,k out.png`.
3. `fuse6.py 0.4 160 [bones]` — slice SDFs (IoU with prior >= 0.55) + 'outside' on the first slice beyond each bone, slab forward model, thin-plate reg. LAM 2 looked blobbier; 0.4 kept. `HOLD=301,401` leaves the axial pair out.
4. `mesh6.py phi.pkl 10` — cubic upsample x1.5, marching cubes, Taubin 10 it (faces must stay < 65536 verts: viewer uses Uint16 faces; 16 MB page limit).
5. `measure6.py`, `export6.py` -> data6.js.

Validation (median / mean contour distance, mm): all six used 0.30–0.44 / 0.39–0.50. Axial pair held out, clav+acro rebuilt from 4 series, compared on axial contours: 0.60–0.67 / 0.75–0.91 (= accuracy between measured slices).
Measurements v4: min AC gap 2.6 mm, CC 11.8 mm, inferior step 5.0 mm (v3: 2.7 / 10.9 / 3.4).

## v4.1 (2026-10-11 evening): smoother arcs, full humeral head
- Earlier humerus was only the cap above the physeal scar. `humfit.py` fits a sphere to the upper cortex arcs (c=(-13.3,-38,-10), r=19 mm) -> `lab_humball.npy`; `seg6.py` with `LABF=lab_humball.npy ONLY=hum DIL=4 ERO=3 OUT=seg_hum3.pkl`; fuse (`SEGF=seg_hum3.pkl SMOOTH=1.0`, LAM 1) -> new prior -> re-seg `DIL=5 OUT=seg_hum4.pkl` -> final `SEGF=seg_hum4.pkl MINIOU=0.8 SMOOTH=1.5 fuse6.py 1.5 160 hum`.
- Other bones: `SMOOTH=1.2 fuse6.py 1.0 160` (2D SDF Gaussian-smoothed before fusion).
- Mesh: `GS=0.6 GS_hum=1.5 TAG=f mesh6.py phi6_d.pkl 20` (3D Gaussian on phi, Taubin 20). Export `TAG=f measure6.py; TAG=f export6.py`; template `tpl7.html`.
- Contour distance (median): clav/acro/cora 0.42 mm, hum 0.67 mm (smoothing trade-off). Measurements: AC gap 3.6, CC 12.7, inferior step 4.6 mm (smoothing shrinks tips ~1 mm).
