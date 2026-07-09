# RMD-X

RMD-X is an external extension area for RAXO experiments. Code in this package
must preserve the RAXO baseline by default:

- read existing RAXO files without mutating them;
- write summaries and experimental outputs under `outputs_rmdx/`;
- keep reusable scripts under `scripts_rmdx/`;
- do not modify `raxo/*.py` or official run scripts.

Batch7 defines contracts and validators only. It does not implement reliability,
material, or risk scoring.

