"""Reading the MR series and writing anonymised copies."""
import glob, os
import numpy as np
import pydicom
from .geometry import M, to_view

# header fields that identify the patient, the site or the visit
_BLANK = ['PatientName', 'PatientID', 'PatientBirthDate', 'PatientBirthTime', 'PatientAge', 'PatientSex', 'PatientWeight',
          'PatientSize', 'PatientAddress', 'PatientTelephoneNumbers', 'OtherPatientIDs', 'OtherPatientNames',
          'PatientComments', 'AccessionNumber', 'StudyID', 'InstitutionName', 'InstitutionAddress',
          'InstitutionalDepartmentName', 'ReferringPhysicianName', 'PerformingPhysicianName', 'NameOfPhysiciansReadingStudy',
          'OperatorsName', 'RequestingPhysician', 'StationName', 'DeviceSerialNumber', 'RequestedProcedureID',
          'PerformedProcedureStepID', 'ScheduledProcedureStepID', 'StudyDescription', 'RequestAttributesSequence',
          'ReferencedPatientSequence', 'ReferencedStudySequence', 'ReferencedPerformedProcedureStepSequence']


def anonymise(src_dir, dst_dir):
    """Copy every DICOM in src_dir to dst_dir/<series>_<instance>.dcm with identifying fields removed.
    Pixel data, geometry and acquisition parameters are untouched. Dates and times are blanked, private tags dropped,
    UIDs replaced by fresh ones derived consistently per original UID."""
    os.makedirs(dst_dir, exist_ok=True)
    uid_map = {}
    def new_uid(u):
        if u not in uid_map:
            uid_map[u] = pydicom.uid.generate_uid(entropy_srcs=['acmr', u])
        return uid_map[u]
    n = 0
    for f in sorted(glob.glob(os.path.join(src_dir, '*'))):
        try:
            d = pydicom.dcmread(f)
        except Exception:
            continue
        d.remove_private_tags()
        for t in _BLANK:
            if t in d:
                d.data_element(t).value = 'ANON' if t in ('PatientName', 'PatientID') else ''
        def walk(ds):
            for el in ds:
                if el.VR == 'SQ':
                    for it in el.value:
                        walk(it)
                elif el.VR in ('DA', 'DT', 'TM'):
                    el.value = ''
                elif el.VR == 'PN':
                    el.value = 'ANON'
                elif el.VR == 'UI' and el.tag not in (0x00020010, 0x00080016, 0x00020002) and el.value:
                    el.value = new_uid(str(el.value))
        walk(d)
        if hasattr(d, 'file_meta') and 'MediaStorageSOPInstanceUID' in d.file_meta:
            d.file_meta.MediaStorageSOPInstanceUID = d.SOPInstanceUID
        d.PatientIdentityRemoved = 'YES'
        d.DeidentificationMethod = 'acmr: ids, names, dates, private tags removed; new UIDs'
        name = f"{int(d.SeriesNumber):04d}_{int(d.InstanceNumber):03d}.dcm"
        d.save_as(os.path.join(dst_dir, name), enforce_file_format=True)
        n += 1
    return n


def _index(dicom_dir):
    out = {}
    for f in glob.glob(os.path.join(dicom_dir, '*')):
        try:
            d = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        out.setdefault(str(d.SeriesNumber), []).append(f)
    return out


def load_stack(dicom_dir, ser, inplane=0.3, _cache={}):
    """One series as dict(img[k, row, col], pos[k], r, c, n, sp, th, gap) in viewer coordinates (stack's own frame).
    inplane: target pixel size; pixels are block-averaged by round(inplane / native) (1 for all diagnostic series)."""
    key = (os.path.abspath(dicom_dir))
    if key not in _cache:
        _cache[key] = _index(dicom_dir)
    ds = [pydicom.dcmread(f) for f in _cache[key][ser]]
    ds.sort(key=lambda d: int(d.InstanceNumber))
    o = np.array(ds[0].ImageOrientationPatient, float)
    r, c = o[:3] @ M.T, o[3:] @ M.T
    n = np.cross(r, c)
    ps = float(ds[0].PixelSpacing[0])
    f = max(1, int(round(inplane / ps)))
    imgs, pos = [], []
    for d in ds:
        a = d.pixel_array.astype(np.float32)
        R = (a.shape[0] // f) * f
        a = a[:R, :R].reshape(R // f, f, R // f, f).mean((1, 3))
        imgs.append(a)
        pos.append(to_view(np.array(d.ImagePositionPatient, float)) + (r + c) * ps * (f - 1) / 2)
    pos = np.array(pos)
    z = pos @ n
    if np.median(np.diff(z)) < 0:
        imgs, pos, z = imgs[::-1], pos[::-1], z[::-1]
    return dict(ser=ser, img=np.stack(imgs), pos=np.asarray(pos), r=r, c=c, n=n, sp=ps * f,
                th=float(ds[0].SliceThickness), gap=float(np.median(np.diff(z))))


def load_all(dicom_dir, series=None, inplane=0.3):
    from .geometry import SERIES
    return {s: load_stack(dicom_dir, s, inplane) for s in (series or SERIES)}
