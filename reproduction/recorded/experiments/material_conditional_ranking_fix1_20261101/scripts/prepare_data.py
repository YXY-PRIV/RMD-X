#!/usr/bin/env python3
"""Explicit grouped-data build, or source inspection without output writes."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.group_data import inspect_only, prepare, read, ref

parser = argparse.ArgumentParser()
parser.add_argument('--protocol', required=True)
parser.add_argument('--protocol-sha256', required=True)
parser.add_argument('--inspect-only', action='store_true')
args = parser.parse_args()
assert ref(args.protocol)['sha256'] == args.protocol_sha256
result = inspect_only(read(args.protocol)) if args.inspect_only else prepare(args.protocol, args.protocol_sha256)
print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
