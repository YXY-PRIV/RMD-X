#!/usr/bin/env python3
"""Source-only portable cached replay wrapper; inspect is the default.

No evaluator, image decoder, model, feature extraction, parameter search or AP
implementation is present. Heavy libraries are imported only by --compute.
The original rmdx files must be staged separately, without editing their bytes.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys


class ContractError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def small_source_sha(path):
    """Used only for small original Python sources, never tensor payloads."""
    require(path.is_file() and path.stat().st_size <= 262144,
            f"Missing or unexpectedly large source file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_ref(ref, roots):
    require(isinstance(ref, dict) and set(ref) == {"root", "path"}, "Invalid path reference")
    require(ref["root"] in roots, "Unknown path root")
    relative = Path(ref["path"])
    require(bool(ref["path"]) and not relative.is_absolute() and ".." not in relative.parts,
            "Configuration paths must be relative and contain no '..'")
    # Explicit roots can themselves point at mounted local artifacts. The wrapper
    # does not copy, discover, download or claim public availability of artifacts.
    return roots[ref["root"]] / relative


def load_rows(path):
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    else:
        rows = read_json(path)
    require(isinstance(rows, list), f"Expected an ordered row list: {path}")
    return rows


def checked_config(config, contract):
    require(config.get("schema_version") == 1, "Unsupported configuration version")
    require(config.get("mode_default") == "inspect", "Default must be inspect")
    profile_name = config.get("arithmetic_profile")
    require(profile_name in contract["profiles"], "Unknown arithmetic/source profile")
    profile = contract["profiles"][profile_name]
    for key in ("family", "shape", "lambda_material", "negative_columns"):
        require(config.get(key) == profile[key], f"Profile mismatch: {key}")
    require(config.get("unc") == {"field": "dist_cosine_inter", "operator": ">", "threshold": 0.15},
            "Only strict native dist_cosine_inter > 0.15 is supported")
    require(config.get("full_J") is True and config.get("replace_score_with_fused") is False,
            "Full J and unchanged raw detector score are mandatory")
    require(set(config["inputs"]) == {"visual", "material_class", "proposals", "categories", "prototypes"},
            "Unexpected or missing input roles")
    require(config["formats"]["visual"] in ("npy", "pt") and
            config["formats"]["material_class"] in ("npy", "pt"), "Unsupported matrix format")
    require(config["dtype"] == "float32", "Accepted families use float32 matrices")
    if profile.get("compute_allowed", True):
        require(config.get("expected_endpoints") == profile["expected_endpoints"],
                "Expected endpoints differ from the accepted contract")
    return profile


def inspect_metadata(config, roots, profile):
    paths = {key: resolve_ref(ref, roots) for key, ref in config["inputs"].items()}
    categories = load_rows(paths["categories"])
    prototypes = load_rows(paths["prototypes"])
    proposals = load_rows(paths["proposals"])
    shape = config["shape"]
    require(len(categories) == shape["C"] and len(prototypes) == shape["J"] and
            len(proposals) == shape["N_all"], "Metadata counts differ from fixed full-cache shape")
    category_ids, category_names = set(), set()
    for index, category in enumerate(categories):
        require(category["class_index"] == index, "Category columns must retain cache order")
        require(type(category["category_id"]) is int and isinstance(category["category_name"], str),
                "Category IDs/names must have their native types")
        require(category["category_id"] not in category_ids and category["category_name"] not in category_names,
                "Duplicate category ID or exact category name")
        category_ids.add(category["category_id"])
        category_names.add(category["category_name"])
    positive_coverage, negative_count = set(), 0
    for index, proto in enumerate(prototypes):
        require(proto["prototype_index"] == index, "Prototype columns must retain complete original J order")
        canonical = proto["canonical_index"]
        require(type(canonical) is int and 0 <= canonical < len(prototypes), "Invalid canonical index")
        if proto.get("is_negative") is True:
            negative_count += 1
            require(proto.get("material_class_index") is None and proto.get("category_name") is None,
                    "Negative columns require no category material lookup")
            continue
        require(proto.get("is_negative") is False, "Explicit native is_negative metadata is required")
        ci = proto.get("material_class_index")
        require(type(ci) is int and 0 <= ci < len(categories), "Partial material coverage is unsupported here")
        category = categories[ci]
        require(proto["category_name"] == category["category_name"] and
                proto["category_id"] == category["category_id"],
                "Exact category_name / category ID / class column conflict; no runtime aliases")
        canonical_proto = prototypes[canonical]
        require(not canonical_proto.get("is_negative") and
                canonical_proto.get("category_name") == category["category_name"] and
                canonical_proto.get("category_id") == category["category_id"],
                "Canonical column must belong to the same positive category")
        positive_coverage.add(ci)
    require(negative_count == config["negative_columns"], "Negative column count mismatch")
    require(positive_coverage == set(range(len(categories))), "Every material category needs a positive prototype")
    for index, row in enumerate(proposals):
        require(row["cache_index"] == index, "Proposal row order must equal the original global cache order")
        require(type(row["image_id"]) is int, "Image IDs must be integers (zero can be legal)")
        require(len(row["bbox"]) == 4 and all(math.isfinite(float(x)) for x in row["bbox"]), "Invalid bbox")
        require(math.isfinite(float(row["detector_score"])), "Invalid raw detector score")
    selected_indices = list(range(len(proposals)))
    id_map = None
    selected_image_count = len({row["image_id"] for row in proposals})
    selection = config.get("role_selection")
    if selection is not None:
        roles_path = resolve_ref(selection["roles"], roots)
        with roles_path.open(newline="", encoding="utf-8") as stream:
            role_rows = list(csv.DictReader(stream))
        all_original_ids = set()
        role_and_local_ids = set()
        id_map = {}
        for row in role_rows:
            original_id, local_id = int(row["original_image_id"]), int(row["COCO_image_id"])
            require(original_id not in all_original_ids, "Original image ID must identify a unique role")
            require(local_id > 0 and (row["role"], local_id) not in role_and_local_ids,
                    "Role-local COCO IDs must be positive and unique within their own namespace")
            all_original_ids.add(original_id)
            role_and_local_ids.add((row["role"], local_id))
            if row["role"] == selection["role"]:
                id_map[original_id] = local_id
        require(len(id_map) == selection["expected_images"], "Selected role image count mismatch")
        for row in proposals:
            require(row.get("original_image_id") == row["image_id"] and row["image_id"] in all_original_ids,
                    "Proposal original-ID namespace does not agree with roles.csv")
        selected_indices = [i for i, row in enumerate(proposals) if row["original_image_id"] in id_map]
        selected_image_count = len(id_map)  # Includes role images without retained proposals.
    require(len(selected_indices) == shape["N_selected"], "Selected proposal count mismatch")
    report = {
        "action": "inspect", "compute_performed": False, "AP_performed": False,
        "family": config["family"], "arithmetic_profile": config["arithmetic_profile"],
        "metadata_checks_passed": True, "matrix_payload_loaded": False,
        "endpoint_signature_verified": False,
        "raw_input_sha256_verified": False,
        "all_intermediate_states_verified": False,
        "all_proposals": len(proposals), "selected_proposals": len(selected_indices),
        "selected_role_images": selected_image_count, "categories": len(categories),
        "full_J": len(prototypes), "negative_columns": negative_count,
        "matrices": {key: {"path": str(paths[key]), "present": paths[key].is_file(),
                            "payload_not_loaded": True} for key in ("visual", "material_class")},
        "compute_allowed_by_profile": profile.get("compute_allowed", True),
        "note": "Metadata-only inspection does not validate tensor values or reproduce an endpoint.",
    }
    return report, paths, proposals, categories, prototypes, selected_indices, id_map


def verified_native_helpers(config, contract, roots):
    helper_root = resolve_ref(config["native_helper_root"], roots).resolve()
    for module, record in contract["native_modules"].items():
        path = helper_root / record["relative_path"]
        require(small_source_sha(path) == record["sha256"], f"Original helper SHA mismatch: {module}")
    # An existing foreign rmdx import is an error, never silently reused.
    for name in ("rmdx", *contract["native_modules"]):
        if name in sys.modules:
            origin = getattr(sys.modules[name], "__file__", None)
            require(origin is not None and Path(origin).resolve().is_relative_to(helper_root),
                    f"Foreign module already loaded: {name}")
    sys.path.insert(0, str(helper_root))
    for module, record in contract["native_modules"].items():
        loaded = importlib.import_module(module)
        require(Path(loaded.__file__).resolve() == (helper_root / record["relative_path"]).resolve(),
                f"Actual import origin mismatch: {module}")
    return importlib.import_module("rmdx.cache_fusion")


def verify_historical_source(config, roots, profile):
    record = profile["raw_source_proof"]
    require(sys.implementation.name == "cpython" and
            f"{sys.version_info.major}.{sys.version_info.minor}" == record["function_AST_python_minor"],
            "Historical compute requires canonical CPython 3.10 AST serialization; metadata inspection remains available")
    path = resolve_ref(config["historical_producer_source"], roots)
    require(small_source_sha(path) == record["sha256"], "Historical producer source SHA mismatch")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    require(record.get("class") == "Runner" and record["function"] == "compute" and
            record.get("qualified_function") == "Runner.compute", "Historical source requires exact Runner.compute identity")
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == record["class"]]
    require(len(classes) == 1, "Historical producer class Runner not found uniquely")
    functions = [node for node in classes[0].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == record["function"]]
    require(len(functions) == 1, "Historical direct method Runner.compute not found uniquely")
    digest = hashlib.sha256(ast.dump(functions[0], include_attributes=False).encode()).hexdigest()
    require(digest == record["function_AST_sha256"], "Historical producer function AST mismatch")
    return {"path": str(path), **record}


def load_matrix(path, format_name, expected_shape, torch):
    if format_name == "npy":
        import numpy as np
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        require(matrix.dtype == np.dtype("float32") and matrix.shape == tuple(expected_shape),
                f"NPY dtype/shape mismatch: {path}")
        # Own the buffer: a read-only mmap is not mutated or passed as writable.
        tensor = torch.from_numpy(matrix.copy())
    else:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        require(isinstance(tensor, torch.Tensor), "PT payload must be a plain tensor")
        require(tensor.dtype == torch.float32 and tuple(tensor.shape) == tuple(expected_shape),
                f"PT dtype/shape mismatch: {path}")
    require(tensor.device.type == "cpu" and bool(torch.isfinite(tensor).all()), "Non-finite or non-CPU matrix")
    return tensor


def ordered_signature(rows):
    digest = hashlib.sha256()
    for row in rows:
        token = [int(row["image_id"]), int(row["category_id"]),
                 *(float(value).hex() for value in row["bbox"]), float(row["score"]).hex()]
        digest.update((json.dumps(token, separators=(",", ":")) + "\n").encode())
    return {"count": len(rows), "sha256": digest.hexdigest()}


def compute(config, roots, contract, profile, prepared, mode):
    require(profile.get("compute_allowed", True), "The synthetic metadata fixture cannot compute")
    report, paths, proposals, categories, prototypes, indices, id_map = prepared
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if config["arithmetic_profile"] == "historical_P02_raw":
        report["historical_source_verified"] = verify_historical_source(config, roots, profile)
    native = verified_native_helpers(config, contract, roots)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    shape = config["shape"]
    visual = load_matrix(paths["visual"], config["formats"]["visual"], [shape["N_all"], shape["J"]], torch)
    material = load_matrix(paths["material_class"], config["formats"]["material_class"],
                           [shape["N_all"], shape["C"]], torch)
    if len(indices) != shape["N_all"]:
        selection = torch.tensor(indices, dtype=torch.long)
        visual, material = visual[selection], material[selection]
    cache = {"visual_similarity": visual, "material_class_similarity": material,
             "proposal_meta": [proposals[index] for index in indices],
             "category_meta": categories, "prototype_meta": prototypes}
    weight = 0.0 if mode == "visual" else float(config["lambda_material"])
    native_mode = "visual_only" if mode == "visual" else "material_preargmax"
    with torch.inference_mode():
        if mode == "additive" and config["arithmetic_profile"] == "historical_P02_raw":
            # Exact accepted P02 compute expression; separate float32 multiply/add.
            # Do not replace this with the algebraically equivalent centered path.
            material_proto = native.material_class_to_prototype(material, prototypes)
            fused = visual + float(weight) * material_proto
            gate = torch.ones(visual.shape[0], dtype=torch.float32)
        else:
            fused, material_proto, gate = native.fuse_from_cache(
                cache, mode=native_mode, lambda_material=weight, lambda_reliability=0.0)
        native_rows = native.detections_from_fusion(
            cache, fused, material_proto, gate, mode=native_mode,
            lambda_material=weight, lambda_reliability=0.0, replace_score_with_fused=False)
    # The original exporter does full-J argmax, negative rejection and canonical
    # diagnostics. This wrapper only applies the accepted strict UNC comparison,
    # preserves exporter order and emits its unchanged bbox and detector score.
    output = []
    for row in native_rows:
        if float(row["dist_cosine_inter"]) > 0.15:
            image_id = int(row["image_id"])
            if id_map is not None:
                require(image_id in id_map, "Exported row escaped the selected role")
                image_id = id_map[image_id]
            output.append({"image_id": image_id, "category_id": int(row["category_id"]),
                           "bbox": row["bbox"], "score": row["score"]})
    signature = ordered_signature(output)
    require(signature == config["expected_endpoints"][mode],
            f"Endpoint mismatch; nothing exported: actual={signature}, expected={config['expected_endpoints'][mode]}")
    report.update(action="compute", compute_performed=True, mode=mode, AP_performed=False,
                  matrix_payload_loaded=True, endpoint_signature_verified=True,
                  raw_input_sha256_verified=False, all_intermediate_states_verified=False,
                  endpoint_signature=signature, lambda_material=weight,
                  native_helper_sha_verified=True, raw_box_and_score_preserved=True,
                  full_J=shape["J"], role_mapping_applied=id_map is not None)
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--compute", action="store_true", help="Explicitly load tensors and replay; never evaluates AP")
    parser.add_argument("--mode", choices=("visual", "additive"), default="visual")
    parser.add_argument("--output", type=Path, help="Required only for explicit compute; existing files are not overwritten")
    args = parser.parse_args()
    roots = {"data": args.data_root.resolve(), "artifact": args.artifact_root.resolve(),
             "bundle": args.bundle_root.resolve()}
    config = read_json(args.config)
    contract = read_json(resolve_ref(config["native_contract"], roots))
    profile = checked_config(config, contract)
    prepared = inspect_metadata(config, roots, profile)
    if not args.compute:
        require(args.output is None, "Inspect writes no files; redirect stdout if a receipt is desired")
        print(json.dumps(prepared[0], indent=2, ensure_ascii=False))
        return
    require(args.output is not None, "--compute requires --output")
    output = args.output
    receipt = output.with_suffix(output.suffix + ".receipt.json")
    require(not output.exists() and not receipt.exists(), "Refusing to overwrite an existing output or receipt")
    require(output.parent.is_dir(), "Create the output directory explicitly before computing")
    rows, report = compute(config, roots, contract, profile, prepared, args.mode)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(rows, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ContractError, KeyError, ValueError, OSError) as error:
        print(f"REPLAY_REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
