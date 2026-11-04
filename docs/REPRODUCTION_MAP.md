# Runtime map

- Segmentation model: midas/model.py, the retained ASFormer encoder/decoders.
- State inference: core/smoothing.py and core/fast_smoothing.py.
- Segmentation commands: python -m midas.
- Segmentation preparation: python -m midas.prepare.
- Localization data/readout/AP: midas/localization.py.
- Localization commands: python -m midas.tal.
- SurgVU reconstructed recipe: configs/surgvu24/midas.json.
- Custom 50Salads paired recipes: configs/50salads.

All portable commands depend only on this repository and declared dependencies.
Historical diagnostic scripts and competing-method training projects are not
runtime dependencies. Do not run old parent-workspace code generators over this
maintained implementation.
