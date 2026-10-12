# AC-ízület 3D rekonstrukció

Jobb váll AC-ízület MR-vizsgálatának interaktív 3D nézője (Philips Ingenia 1,5 T, 6 diagnosztikai sorozat, 3,5 mm-es rétegek).

**Élő oldal:** https://sannyer.github.io/ac-mr-3d/

A nézet minden beállítása (rétegek, áttetszőség, ablakolás, metszősík, eredeti szelet, kamera) a link query stringjében van. A „Link a nézethez” gomb a mostani nézet linkjét a vágólapra másolja, így egy adott nézet linkben elküldhető.

Nem radiológiai lelet.

## Tartalom

| Hely | Mi van benne |
|---|---|
| `index.html` | a néző (three.js r128 CDN-ről); az adatot a `data/viewer/ac3d_data.js` fájlból tölti |
| `data/dicom/` | az eredeti felvétel 181 DICOM-fájlja, **anonimizálva** (név, születési adat, azonosítók, intézmény, dátumok, UID-k, privát mezők törölve; a képadat és a geometria érintetlen) |
| `data/derived/` | a generált adatok: szuperfelbontású térfogatok, csontmezők, címkék, kontúrok, hálók, mérések ([formátum](data/README.md)) |
| `data/viewer/ac3d_data.js` | a néző adatcsomagja |
| `acmr/` | a generátor (Python csomag) |

## Generálás

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # vagy: pip install -e .
pip install torch                      # opcionális, GPU-hoz (NVIDIA-hoz a pytorch.org CUDA-s parancsa)
python -m acmr run                     # data/dicom -> data/derived + data/viewer
python -m acmr run --from mesh         # csak egy szakasztól (a köztes eredmények a work/ mappában)
```

Szakaszok: `srr` → `contours` → `humerus` → `mesh` → `export`. Négymagos CPU-n nagyjából egy óra (ebből a PD szuperfelbontás kb. 25 perc), GPU-val ennek töredéke.

Kézzel rögzített bemenetek: a kiinduló csont-magpontok (`acmr/seeds.py`, a 601-es és 801-es sorozaton) és a humerusfej-gömb (`acmr/humerus.py: SPHERE`). Minden más a DICOM-okból számolódik.

A futás felülírja a `data/viewer/ac3d_data.js`-t. A repóban most lévő változat a korábban közzétett modell; a nulláról generált eredmény ettől kicsit eltér: a csontok Dice-egyezése 0,84–0,98, AC-rés 3,1 mm (korábban 3,6), CC 11,3 mm (12,7). A zsírfényes térfogat új módszerrel készül (a két sorozat külön, a PD éleivel vezetve, majd átlagolva): rácsműtermék nincs benne, kicsit lágyabb a korábbinál.

GPU: ha a PyTorch telepítve van és van CUDA- vagy Apple MPS-eszköz, a nehéz lineáris algebra (vetítő modell, regularizálás, konjugált gradiens) azon fut; különben NumPy CPU-n. Kényszeríteni az `ACMR_DEVICE` változóval lehet: `auto` (alapértelmezett), `cuda`, `mps`, `cpu-torch`, `numpy`. A 2D kontúrkeresés CPU-magokon párhuzamos (`ACMR_PROCS`).

## Módszer röviden

1. **Szuperfelbontás (`srr`)**: minden sorozat minden pixele a 3,5 mm-es réteg vastagságán átlagolt 3D térfogat (szelet-modell, sorozatonkénti merev regisztráció, sorozatonkénti sima bias-mező). A PD térfogat a három irányú PD sorozatból Huber-TV regularizálással készül (0,5 mm-es rács, ±45 mm az AC-ízület körül). A zsírfényes térfogat (T2 koronális + T1 szagittális) simítása a PD élein is megáll, így eltűnik a kétirányú rácsműtermék.
2. **Kontúrok (`contours`)**: random walker 2D csontkontúr a natív szeleteken. Az első kör kézzel olvasott magpontokból indul (601, 801), a következő két kör a 3D modell vetületéből, mind a 6 sorozat minden érintett szeletén.
3. **Fúzió**: a szeletkontúrok csonkolt előjeles távolságmezői ugyanazon a szelet-modellen át egy 3D mezővé állnak össze (vékonylemez-regularizálás, „kívül” jelzés a csont utolsó szelete utáni szeleten).
4. **Felkarcsont (`humerus`)**: a fej felső kortikális íveire illesztett gömb ad előzetes alakot, ezzel két újabb kontúrkör és fúzió.
5. **Háló és mérés (`mesh`)**: enyhe 3D simítás, köbös felmintavételezés, marching cubes, Taubin-simítás; AC-rés, CC-távolság, lépcső.
6. **Export**: `data/derived` és a néző adatcsomagja.
