#!/usr/bin/env python3
"""Explicit CPU-only auxiliary bank alignment; requires frozen conductor gates."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.align_data import prepare

parser = argparse.ArgumentParser()
parser.add_argument('--protocol', required=True)
parser.add_argument('--protocol-sha256', required=True)
args = parser.parse_args()
print(json.dumps(prepare(args.protocol, args.protocol_sha256), ensure_ascii=False), flush=True)
