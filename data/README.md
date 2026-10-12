# Adatformátumok

Koordináták (minden generált adatban): mm, **x = beteg bal oldala felé, y = felfelé (superior), z = előre (anterior)**, origó az AC-ízület közepén, ami DICOM LPS-ben (-123,415, -40,525, 63,645) mm. A rács ±45 mm-es kocka 0,5 mm-es voxelekkel: 180³, tömbindex `[x, y, z]`, a voxel középpontja `(i + 0,5) · 0,5 − 45`. A térfogatok a 301-es (PD axiális) sorozat keretében vannak; a többi sorozat merev regisztrációja az `acmr/geometry.py` `REG` táblájában van.

## `dicom/`
`SSSS_III.dcm` = sorozatszám (4 jegy) és szeletszám. Anonimizálva (`acmr/dicomio.py: anonymise`).

| Sorozat | Leírás |
|---|---|
| 301 | PD zsírelnyomott, axiális |
| 401 | T2 zsírelnyomott, axiális |
| 501 | PD zsírelnyomott, koronális |
| 601 | T2, koronális |
| 701 | PD zsírelnyomott, szagittális |
| 801 | T1, szagittális |

## `derived/`
- `volumes.npz`: `pd`, `fat` — float16, 180³, 0–1 (99,5. percentilisre normálva). `pd`: három irányú PD szuperfelbontás; `fat`: zsírfényes (T2 kor. + T1 szag.) a PD éleivel vezetve.
- `bones.npz`: `labels` — uint8, 180³: 0 háttér, 1 ízület körüli lágyrész-zóna, 2 kulcscsont, 3 acromion, 4 coracoid, 5 felkarcsont. `sdf_clav`, `sdf_acro`, `sdf_cora`, `sdf_hum` — float16 előjeles távolság mm-ben (negatív = csonton belül, ±10-re vágva), a hálók ezek 0-szintjei.
- `contours.npz`: a fúzióba ment 2D szeletkontúrok. Kulcs `SER_KK_bone` (pl. `601_12_acro`): `np.packbits` maszk, alakja a `..._shape` kulcsban; a maszk a natív szelet pixelrácsán van (sor, oszlop).
- `meshes/*.ply`: bináris PLY háromszöghálók, mm-ben a fenti keretben.
- `measurements.json`: legkisebb AC-rés, CC-távolság, felső/alsó lépcső, a mért pontok és a humerusfej-gömb. Geometriai mérés, nem lelet.

## `viewer/ac3d_data.js`
`window.AC3D = {...}`: a néző kivágott (157×134×113 voxel) uint8 térfogatai base64-ben `[z][y][x]` sorrendben (`pd`, `fat`), futáshossz-kódolt címkék (`labrle`, `nearrle`: uint32 párok érték, hossz), hálók (`v` float32, `f` uint16), mérések, és sorozatonként az eredeti szeletek JPEG-atlasza (`natives`: 84 mm-es kivágás, 240 px; `o` a csempe sarka, `r`, `c` a sor/oszlop irány a referencia-keretben).
