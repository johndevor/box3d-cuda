# Portable, unchanged robot reference inputs

These files are copied byte-for-byte from the original accepted local model
translation inputs. They contain model/physics data, not credentials or
machine-specific runtime binaries. SHA-256 is enforced by
`experimental/contact_v1/model_translation.py` before use:

| File | SHA-256 |
|---|---|
| `open-duck-native-compat-v1/geometry-goldens.json` | `e52ba7d0f79434499d8fb6c2d611eb46ee12e2f32cb36258b38cd22959d0b08b` |
| `open-duck-zero-hold-cpu-v1/cpu-result.json` | `a6d578064b433e730612d7144742b706471e63a37e3c81bcbc24acb7a7203a58` |
| `open-duck-zero-hold-cpu-v1/model/open_duck_mini_v2.xml` | `968b18de4e3f55b31252155f52779fa490989f5da92bc9b308e0bb4e81d6bb5c` |

The reference is a historical CPU/MuJoCo zero-action record, not a trained
policy or evidence of native/CUDA walking. No reference values, model masses,
inertias, targets, geometry or physical tolerances were changed by packaging.

Model provenance: [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini).
Retained notices/licenses are in `../model/asset-NOTICE`,
`../model/asset-LICENSE`, and `../model/hardware-LICENSE`. The XML is the pinned
simulation candidate (including the documented local collision/reset edits),
not a claim that all upstream CAD files are redistributed here unmodified.
