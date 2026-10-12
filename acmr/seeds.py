"""Hand-read seed points for the initial bone contours on 601 (coronal T2) and 801 (sagittal T1): local mm in a +-30 mm crop
around the AC-centre projection (origin top-left), plus slices where a bone is known to be absent."""
SEEDS = {
    '801': {
        'clav': {10: [(6, 22)], 11: [(14, 20)], 12: [(21, 21)], 13: [(28, 19)], 14: [(31, 20)], 15: [(29, 22)], 16: [(27, 26)], 17: [(25, 28)]},
        'acro': {14: [(53, 30)], 15: [(52, 31)], 16: [(50, 31)], 17: [(42, 35)], 18: [(42, 36)], 19: [(36, 37)]},
        'hum': {14: [(40, 66)], 15: [(40, 64)], 16: [(38, 62)], 17: [(35, 57)], 18: [(35, 57)], 19: [(35, 58)]},
    },
    '601': {
        'clav': {10: [(55, 23)], 11: [(55, 22)], 12: [(45, 27)], 13: [(46, 25)], 14: [(50, 23)]},
        'acro': {13: [(19, 35)], 14: [(25, 35)], 15: [(25, 37)], 16: [(30, 39)], 17: [(40, 42)], 18: [(45, 48)]},
        'hum': {10: [(25, 66)], 11: [(20, 63)], 12: [(20, 60)], 13: [(20, 58)], 14: [(20, 62)]},
        'cora': {7: [(50, 52)], 8: [(50, 48)], 9: [(57, 47)], 10: [(62, 46)]},
    },
}
ABSENT = {
    '801': {'clav': [18, 19, 20], 'acro': [21, 22]},
    '601': {},
}
PARAMS = {
    'clav': dict(rel=0.55, R=22, skin=3.0, open_mm=0.7, cortex_mm=1.0),
    'acro': dict(rel=0.55, R=26, skin=3.0, open_mm=0.7, cortex_mm=1.0),
    'cora': dict(rel=0.6, R=14, skin=3.0, open_mm=0.7, cortex_mm=1.0),
    'hum': dict(rel=0.6, R=30, skin=3.0, open_mm=1.0, cortex_mm=1.0),
}
OVR = {}  # per-slice parameter overrides: {'801:16:clav': {'rel': 0.4}}
BG = {'801:16': [(17, 21)]}  # extra background points (local mm)
