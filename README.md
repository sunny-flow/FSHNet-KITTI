# FSHNet KITTI research code — release candidate

This package is a curated snapshot of a locally modified FSHNet/OpenPCDet-derived research project. It contains KITTI-oriented model code and configurations, not an authenticated reproduction of every reported experiment. The upstream project description and attribution are retained in [README_UPSTREAM.md](README_UPSTREAM.md); upstream benchmark results there are not results of this modified version.

**Status: privately prepared release candidate.** Public access, a clean installation, full training, and benchmark reproduction have not yet been independently validated. Verify provenance and third-party rights before publishing. No checkpoints or raw datasets are included. The KITTI split identifier lists are retained for inspection.

## Contents

- `pcdet/`: model, data processing, evaluation and CUDA/C++ extension source.
- `tools/`: training/evaluation entry points and configurations from the supplied archive.
- `tests/`, `docs/`, `docker/`: retained source/documentation from the archive.
- `release_notes/`: archive-source manifest, the documented optimizer change and validation scope.

The module implementations can be inspected in `pcdet/models/backbones_3d/fshnet_light.py` (RPDI), `pcdet/models/dense_heads/sparse_dynamic_head.py` (SN-DSLA and GCQC integration), and `pcdet/models/model_utils/sparse_geometry_utils.py` (GCQC implementation). Presence of a switch or a tensor does not establish that the module was effective in a historical checkpoint.

## Provenance and important limitations

Extracted from the author's `FSHNet-main.zip` (SHA-256 in `release_notes/PROVENANCE.md`), excluding datasets, compiled libraries, model weights, outputs, logs and editor caches. The optimizer fix, a Python syntax fix, and path portability adaptations are recorded in `release_notes/release_changes.json`; other retained source files are unchanged. The optimizer originally omitted parameters directly registered on non-leaf modules, including `rpdi_scale`. This candidate substitutes a previously audited fix; the pre-fix source and patch are retained for comparison. **This is a new source version**, not proof that any historical model was trained with the fix. Do not resume an old optimizer state after the parameter structure changes without explicit compatibility verification.

The archive does not supply a verified set of all eight historical ablation configurations or a validated full KITTI training configuration binding the three modules and the published checkpoint. Some configurations contain machine-specific dataset paths or unrelated switches; inspect and adapt them before use. Check effective switches, parameter coverage, split, seed and the checkpoint-selection rule for every run. Historical controls and RPDI attribution remain unresolved.

## Environment and data

`requirements.txt` was retained from the supplied archive as a historical package list; it is not a tested or complete lock file. Versions of Python, PyTorch, CUDA, spconv and torch-scatter must be matched to the target machine, and the C++/CUDA extensions need to be built there. See `README_UPSTREAM.md` and `docs/INSTALL.md` as historical instructions; their commands have not been verified on a clean system for this snapshot.

Obtain KITTI directly under its own terms and prepare the expected data layout and info files as described in `docs/GETTING_STARTED.md` and the project dataset configuration. Update any absolute `DATA_PATH` in your chosen YAML. KITTI train/val/test identifier lists are included under `data/kitti/ImageSets/`; their provenance and redistribution rights still require author confirmation. Raw KITTI files, weights and model outputs are not included.

After environment setup and dataset preparation, the existing entry points are `tools/train.py` and `tools/test.py`. Supply the path to an independently verified config and checkpoint through their command-line arguments; inspect `--help` on the target installation. No paper-number reproduction is asserted by this package.

## Licenses and attribution

The input includes an Apache-2.0 `LICENSE` and retains nested third-party notices and license files where present. Different copied components, external dependencies, datasets and weights may have their own terms. Confirm authorship, modifications, source attributions and redistribution rights for each component before public release.
