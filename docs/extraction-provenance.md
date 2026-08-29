# Extraction provenance

This repository was extracted from the `box3d_cuda/` subtree of
`box3d-arm-lab` with `git subtree split`, preserving the engine file history
from the original Stage 0 through Stage 7 commits. Backend-neutral benchmark,
joint, ray, and coupling contracts were then copied out of their former
Factory OS namespaces at source commit `3e648ac`.

The Box3D-derived mappings remain pinned and enumerated in `UPSTREAM.json`.
The upstream Box3D license is retained verbatim in `LICENSE.box3d`; the
standalone integration code is distributed under `LICENSE`.

The extraction is organizational only. It does not claim a broader Box3D port
or modify accepted physics semantics, thresholds, workloads, or results.
