# Artifact guide

An artifact is a file needed to understand or repeat a result: code, a model, a prediction table or a saved verification record.

| File | What it tells you |
| --- | --- |
| [manifest.json](manifest.json) | Current package file paths, sizes and SHA-256 hashes |
| [source_map.json](source_map.json) | Original workspace paths and hashes; packaged hashes; documented changes |
| [external_artifacts.json](external_artifacts.json) | Required data and remote checkpoints: locations, access, checksums and availability |

Current magnitude forests and the one current prediction file use ordinary Git because they are small. No second model ZIP is included. Original datasets stay outside Git. Large future artifacts should use approved company storage with locations/hashes recorded here; moving them to a service is not authorized by this handoff.

The manifest excludes itself and local .git, .validation, cache and virtual-environment files. No hash is presented as independently verified remote content unless explicitly marked. Historical manifests describe original deliveries, including files outside this selected package. They are not the package manifest.

Intentionally pruned experiments remain represented by compact history. Their old verification does not imply their models/predictions are available. Shared prepared-data storage and colleague permissions remain to be confirmed.
