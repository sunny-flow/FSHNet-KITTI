# Source provenance and verification boundaries

Input ZIP SHA-256: `19766d69a5b80ff741e6605fe513071df2bd65f09b01a26d28cc26af5beae206`. `archive_source_manifest.csv` records SHA-256 of each retained original archive file. The original root README is preserved as README_UPSTREAM.md.

The optimizer file in the input has SHA-256 `aa83865cc3473e9aa5a75c0367921e0a35ad9d7ee411d67637481cb016272113`; the previously tested fix before adding the release notice has SHA-256 `f7ae6aef20abc41cdaac73d1ad9195951b4c8676f1a84dba4fdacfee6fcf2cc5`. The `optimizer_fix.diff` and original optimizer file are provided. Prior server-side audit reported 235/235 parameter tensors covered with 0 duplicates for its specifically audited Full model; this result is not a fresh end-to-end test of this entire release candidate.

The release additionally contains a syntax fix and relative-path adaptations; `release_changes.json` records every modified path. The input ZIP does not carry independently authenticated historical training identities, eight complete historical ablation YAMLs, raw datasets or checkpoints. This repository must not be cited as proof that those items were available to reviewers. The source may also contain remaining environment-specific paths.
