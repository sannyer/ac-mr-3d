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


# Extra points added at the v4 review, where the single hand point above left part of the bone (or the whole slice) out.
# Same units (local mm in the +-30 mm crop). The humeral head is rebuilt from a sphere prior instead (humerus.py).
EXTRA = {
    '601': {
        'clav': {12: [(33.8, 27.9)], 13: [(32.9, 26.7)], 15: [(43.8, 24.2)]},
        'acro': {16: [(47.5, 33.8)], 17: [(43.8, 37.1)]},
        'cora': {8: [(55.4, 47.1)], 9: [(65.4, 42.9)], 10: [(55.0, 40.4)]},
    },
    '801': {
        'clav': {13: [(20.2, 23.2)], 15: [(26.9, 29.2)]},
        'acro': {16: [(52.2, 36.2)], 17: [(56.2, 39.9)], 18: [(56.5, 39.5)], 20: [(36.2, 37.9)]},
        'cora': {10: [(16.2, 42.9)], 11: [(13.5, 43.9)], 12: [(12.5, 44.5)], 13: [(5.2, 47.5)], 14: [(-0.1, 55.2)]},
    },
}
for _s, _bb in EXTRA.items():
    for _b, _kk in _bb.items():
        for _k, _pts in _kk.items():
            SEEDS[_s].setdefault(_b, {}).setdefault(_k, []).extend(_pts)
