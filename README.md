# AC-ízület 3D rekonstrukció

Jobb váll AC-ízület MR-vizsgálatának interaktív 3D nézője (Philips 1,5 T, 6 diagnosztikai sorozat, 3,5 mm-es rétegek).

**Élő oldal:** https://sannyer.github.io/ac-mr-3d/

A nézet minden beállítása (rétegek, áttetszőség, ablakolás, metszősík, eredeti szelet, kamera) a link query stringjében van. A „Link a nézethez” gomb a mostani nézet linkjét a vágólapra másolja, így egy adott nézet linkben elküldhető.

- `index.html`: az önálló néző (three.js r128 CDN-ről, minden adat beágyazva).
- `pipeline/`: a rekonstrukció scriptjei (szuperfelbontású térfogat, csontkontúrok mind a 6 sorozaton, előjeles távolságmezős fúzió, háló). A DICOM-fájlok nincsenek a repóban.

Nem radiológiai lelet.
