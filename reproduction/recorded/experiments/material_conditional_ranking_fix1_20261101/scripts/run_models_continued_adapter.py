#!/usr/bin/env python3
"""Adapt reference dictionaries for the unchanged frozen inference controller.

The initial controller stopped before starting any GPU child because two
metadata reads passed a reference dictionary to pathlib.Path. This adapter
validates those references and supplies their paths; model math stays frozen.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
C=Path(__file__).resolve().parents[1]
O=Path('/mnt/data/yxy-codex/home/outputs/RAXO/material_conditional_ranking_fix1_20261101')
spec=importlib.util.spec_from_file_location('frozen_continuation_controller',C/'scripts/run_models_continued.py')
controller=importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
read_path=controller.read


def read_reference_or_path(value):
    if isinstance(value,dict):
        assert {'path','bytes','sha256'}<=value.keys()
        value=controller.verify(value)
    return read_path(value)


def preflight():
    P,pr=controller.protocol(C/'config/protocol.json',os.environ['RAXO_PROTOCOL_SHA256'])
    controller.actor_gate(pr)
    am,amref,qa=controller.continuation_gate(pr)
    failure=read_reference_or_path(controller.ref(O/'automation/model_failure.json'))
    original_exit=read_reference_or_path(controller.ref(O/'automation/models_service_exit.json'))
    assert failure['status']=='FAILED'
    assert original_exit['observed']['EXIT_STATUS']=='1'
    index=read_reference_or_path(am['retained_inference_index'])
    trained=read_reference_or_path(am['retained_training_complete'])
    assert len(index['records'])==18 and len(trained['runs'])==6
    existing={(r['case'],r['variant'],r['seed']) for r in index['records']}
    planned={(case,v,s) for case in P['inference_cases'] for v in P['variants'] for s in P['seeds']}
    assert existing<=planned and len(planned-existing)==24
    assert not (O/'inference/PIXray/material/seed17').exists()
    return dict(status='PASS_REFERENCE_ADAPTER_METADATA_ONLY',protocol=pr,
        verification_amendment=amref,retained_jobs=18,remaining_jobs=24,
        new_GPU_or_training_or_AP=0,original_controller=controller.ref(C/'scripts/run_models_continued.py'),
        adapter=controller.ref(__file__))


if __name__=='__main__':
    controller.read=read_reference_or_path
    if sys.argv[1:]==['--preflight-only']:
        print(json.dumps(preflight()),flush=True)
    else:
        assert not sys.argv[1:]
        record=preflight()
        controller.new(O/'continuation/driver_adapter_executed.json',dict(
            **record,status='REFERENCE_ADAPTER_ACTUALLY_EXECUTED',PID=os.getpid(),
            started_utc=controller.now(),argv=sys.argv))
        try:
            controller.main()
        except BaseException as exc:
            failure=dict(status='FAILED_CONTINUATION_ADAPTER',failed_utc=controller.now(),error=repr(exc),
                         automatic_retry_allowed=False,new_trainings=0)
            controller.atomic(O/'automation/model_continuation_adapter_failure.json',failure)
            raise
