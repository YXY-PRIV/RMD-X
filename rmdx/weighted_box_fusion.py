"""Simple class-aware Weighted Box Fusion for COCO xywh detections."""

from __future__ import annotations

from typing import Any

from rmdx.ensemble_nms import group_by_image_category, iou_xywh, limit_per_image


def _fuse_cluster(cluster: list[dict[str, Any]], *, score_mode: str) -> dict[str, Any]:
    weights = [max(1e-9, float(d.get("score", 0.0)) * float(d.get("_detector_weight", 1.0))) for d in cluster]
    denom = sum(weights)
    bbox = [sum(float(d["bbox"][i]) * w for d, w in zip(cluster, weights)) / denom for i in range(4)]
    scores = [float(d.get("score", 0.0)) for d in cluster]
    if score_mode == "max":
        score = max(scores)
    elif score_mode == "average":
        score = sum(scores) / len(scores)
    else:
        score = sum(float(d.get("score", 0.0)) * float(d.get("_detector_weight", 1.0)) for d in cluster) / sum(float(d.get("_detector_weight", 1.0)) for d in cluster)
    base = max(cluster, key=lambda d: float(d.get("score", 0.0)))
    out = dict(base)
    out["bbox"] = [round(float(v), 6) for v in bbox]
    out["score"] = float(score)
    out["dist_cosine_inter"] = sum(float(d.get("dist_cosine_inter", 1.0)) * w for d, w in zip(cluster, weights)) / denom
    out["dist_cosine_intra"] = sum(float(d.get("dist_cosine_intra", 0.0)) * w for d, w in zip(cluster, weights)) / denom
    out["ensemble_cluster_size"] = len(cluster)
    out["ensemble_input_detectors"] = sorted({str(d.get("detector", "")) for d in cluster})
    out.pop("_detector_weight", None)
    return out


def weighted_box_fusion(dets: list[dict[str, Any]], *, iou_thr: float, score_thr: float, max_dets_per_image: int, score_mode: str = "weighted_average") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rows in group_by_image_category([d for d in dets if float(d.get("score", 0.0)) >= score_thr]).values():
        pending = sorted(rows, key=lambda r: float(r.get("score", 0.0)) * float(r.get("_detector_weight", 1.0)), reverse=True)
        clusters: list[list[dict[str, Any]]] = []
        while pending:
            seed = pending.pop(0)
            cluster = [seed]
            rest = []
            for row in pending:
                if iou_xywh(seed["bbox"], row["bbox"]) >= iou_thr:
                    cluster.append(row)
                else:
                    rest.append(row)
            clusters.append(cluster)
            pending = rest
        out.extend(_fuse_cluster(cluster, score_mode=score_mode) for cluster in clusters)
    return limit_per_image(out, max_dets_per_image)
