# Current reproduction guide

Start with `python3 reproduction/inspect_release.py` from the repository root. The default inspection is read-only and does not launch training, AP evaluation, bootstrap or GPU workers. The current code and evidence inventory is in [reproduction/README.md](../reproduction/README.md).

The release distinguishes immutable recorded source/protocols from a portable core and external resource configuration. Use the recorded hashes and exact source/validation/tune roles; do not silently substitute proposal families, support banks, annotation-ID conventions or selected checkpoints. Preserve one exported detection per original proposal and the anchor-selected branch/category/box.

Full fitting and native evaluation require the corresponding provider datasets, annotations, upstream models, feature/proposal inputs and recorded environment(s). Follow the resource index and resolve external paths explicitly. Do not treat supplied small-vector checks or source inspection as a fresh reproduction of all recorded AP or raw-image timing.

The original 2026-07 native-stage guide is [archived here](legacy/REPRODUCTION_GUIDE_20260709.md). Its colour-only/centering descriptions and early numeric references are historical and do not define the selected conflict-aware method.
