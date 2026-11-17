# Packaging changes

Original workspace files were not edited, moved or deleted. This repository contains selected copies plus new explanations and wrappers.

| Change | Reason | Validation |
| --- | --- | --- |
| Preserve magnitude playground and timing sibling layout | Keep replay and notebook-builder dependencies together | Dependency reads and inference/source-hash checks |
| Remove earlier notebook outputs, execution counts and nonessential metadata | Avoid publishing embedded company rows or session details | Cell source comparison and notebook output checks |
| Make two earlier feature helpers resolve their root relative to their files | Prevent copied helpers from writing into original C:/ScaleWave paths | Evaluated ROOT assignments without running builders/training |
| Redact personal workstation usernames in six historical metadata/text files | Keep irrelevant personal path details out | Original and changed hashes recorded; scientific values unchanged |
| Add external-data magnitude wrapper | Replay without copying data | Exact original predictions and metrics reproduced |
| Add read-only timing wrapper | Original verifier writes historical progress and receipts | Original checks run in memory; only two writes suppressed; actual frozen source hashes also checked |
| Add .gitattributes with newline normalization disabled | Preserve source hashes across Git checkouts | Git attributes and packaged file hashes checked |

Exact file changes are listed in [source map](../artifacts/source_map.json). All current magnitude replay source and frozen timing source files remain unchanged. Historical embedded paths and old instructions are evidence, not current commands to execute. Original manifests may name excluded/pruned files and pre-promotion versions.

The selected package excludes teammate submission areas, original datasets, caches, large original ZIPs, locks, credential files, personal/NDA documents, unrelated folders, duplicate model ZIP and bulky redundant report versions. No originals were deleted. The older notebook with uncertain ownership was not included.

The original playground README still describes a broader direction/team context. Those shared baseline references are preserved as provenance, not a claim of new direction research or inclusion of unrelated teammates' submissions.
