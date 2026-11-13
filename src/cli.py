"""Run with python -m src.cli --mode both from the repository root."""
import argparse
from .experiments import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',default='data/raw'); p.add_argument('--output',default='results')
    p.add_argument('--mode',choices=['reproduction','strict','both'],default='both')
    p.add_argument('--seeds',type=int,default=20); p.add_argument('--min-tickets',type=int,default=50)
    args=p.parse_args()
    if args.seeds<1: p.error('--seeds must be positive')
    for mode in (['reproduction','strict'] if args.mode=='both' else [args.mode]):
        run(args.raw,args.output,mode,args.seeds,args.min_tickets)


if __name__=='__main__': main()
