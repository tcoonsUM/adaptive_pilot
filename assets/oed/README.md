# OED model assets

These files were retained from the supplied research archive because they are small enough
for normal repository hosting and are required for inference:

- one high-fidelity proxy MLP checkpoint;
- one medium-fidelity four-layer FNO checkpoint;
- one low-fidelity two-layer FNO checkpoint;
- numeric parameters for the fitted one-feature MLP output transform; and
- the MLP output-scaling bounds.

The collaborative archive also contained three fitted identity input transformers and a
scikit-learn `PowerTransformer` serialized with joblib. The public loader represents the
identity transforms directly in code and stores only the numerical Yeo--Johnson power,
standardization mean, and standardization scale in `scalers/mlp_output_transform.npz`. This
avoids executable pickle loading and removes dependence on the scikit-learn version used to
fit the transform.

The raw finite-difference training trajectories are intentionally omitted. The JSON manifest
is the only path configuration consumed by the public model loader.

## Compatibility and trust

PyTorch checkpoints are serialization formats and should only be loaded from a trusted
source. The loader requests `weights_only=True` on supported PyTorch versions and loads only
the three files named in `manifest.json`.

The collaborative archive did not state a license for the checkpoints. The authors must
confirm redistribution permission before publishing the repository and should record the
approved asset license here.
