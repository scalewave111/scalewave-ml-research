# Contributing

Use a research branch and open a pull request for review. See the [GitHub guide](docs/publishing_to_github.md).

Before an experiment, write the question, target, units, feature lookbacks, forecast horizons, dated splits, baseline, population rule, settings, fit budget and stopping rule. After it finishes, record scores, failures, verification type, limitations and artifact paths in the experiment registry.

Keep magnitude and timing environments separate. Never replace a historical result with a newer result under the same identity. Keep original code and frozen source hashes unless a change is explained. Put new experiments in a new clearly named scope and use one active output location.

No company datasets, credentials, personal files, NDA files or unrelated teammate work belong in a commit. Review git diff --cached before pushing. Publication requires the company destination and final content approval. Company ownership/licensing is not inferred; no open-source license is granted by this handoff.
