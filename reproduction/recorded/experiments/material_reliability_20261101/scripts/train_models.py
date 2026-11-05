#!/usr/bin/env python3
"""One frozen-protocol training or label-free inference subprocess."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quality_model import VARIANTS, SEEDS, train, infer


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='mode', required=True)
    for mode in ('train', 'infer'):
        sub = commands.add_parser(mode)
        sub.add_argument('--protocol', required=True)
        sub.add_argument('--protocol-sha256', required=True)
        sub.add_argument('--output-dir', required=True)
        sub.add_argument('--device', default='cuda:0')
        if mode == 'train':
            sub.add_argument('--variant', choices=VARIANTS, required=True)
            sub.add_argument('--seed', type=int, choices=SEEDS, required=True)
        else:
            sub.add_argument('--case', required=True)
            sub.add_argument('--checkpoint', required=True)
    arguments = parser.parse_args()
    result = train(arguments) if arguments.mode == 'train' else infer(arguments)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
