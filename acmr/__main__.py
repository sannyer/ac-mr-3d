"""python -m acmr run [--from STAGE] [--to STAGE]   |   python -m acmr anonymise SRC DST"""
import argparse, os
from . import pipeline, dicomio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(prog='acmr', description='AC-joint MR 3D reconstruction')
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='run the pipeline (stages: ' + ', '.join(pipeline.STAGES) + ')')
    r.add_argument('--dicom', default=os.path.join(ROOT, 'data', 'dicom'))
    r.add_argument('--work', default=os.path.join(ROOT, 'work'), help='cache of intermediate results (~1 GB)')
    r.add_argument('--out', default=os.path.join(ROOT, 'data'))
    r.add_argument('--from', dest='start', choices=pipeline.STAGES)
    r.add_argument('--to', dest='stop', choices=pipeline.STAGES)
    a = sub.add_parser('anonymise', help='copy DICOMs with identifiers removed')
    a.add_argument('src'); a.add_argument('dst')
    args = p.parse_args()
    if args.cmd == 'run':
        pipeline.run(args.dicom, args.work, args.out, args.start, args.stop, log=lambda s: print(s, flush=True))
    else:
        print(dicomio.anonymise(args.src, args.dst), 'files written')


if __name__ == '__main__':
    main()
