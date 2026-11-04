# Upstream provenance

The active ASFormer definitions in midas/asformer.py are extracted from the
local upstream snapshot. Its MIT notice is included in LICENSES/ASFormer.txt.
Device allocation follows the input device; attention arithmetic and learned
parameters are unchanged. Source hashes are in runtime_manifest.json.

MIDAS builds on existing temporal segmentation networks and classical Kalman/RTS inference. This directory does not claim those underlying algorithms or baseline implementations as original.

| Dependency / comparison | Upstream |
| --- | --- |
| ASFormer | https://github.com/ChinaYi/ASFormer |
| FACT | https://github.com/ZijiaLewisLu/CVPR2024-FACT |
| BaFormer | https://github.com/peiyao-w/BaFormer |

C2F-TCN, UVAST and other compared-method training code are not bundled in this
repository. Their local research records remain separate; listing a comparison
does not sublicense its implementation.

Boundary/CDF is a non-official reproduction on C2F-TCN, not an official implementation. The corresponding adapters retain this distinction.

The parent workspace is not distributed. New MIDAS project code uses the MIT
license approved by the author; the upstream ASFormer MIT notice is retained
unchanged and included in the installed package's license data.
