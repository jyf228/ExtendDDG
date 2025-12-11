# ExtendDDG: Expanding Automated Dataset Description Generation with Supplemental Dataset Documentation
ExtendDDG is an extension of [AutoDDG](https://github.com/VIDA-NYU/AutoDDG), developed by the NYU VIDA Lab, to supplement automated dataset description generation with a dataset's associated documentation, such as codebooks, methodology reports, and READMEs.

## Quick Start

### Requirements
- Python >=3.10
- [AutoDDG](https://github.com/VIDA-NYU/AutoDDG)

### Run
```bash
uv sync
```

Run the example Python script.
```bash
cd examples
uv run python quick_start.py
```

## Datasets
We have compiled a set of publicly available datasets that contain supplemental documentation to be used for testing. A summary table of these datasets and their associated documentation can be found [here](https://docs.google.com/spreadsheets/d/1E7oVNsUg7hm7oNZCpTIRuRUhweLwBliA_OymoMmUFEk/edit?gid=0#gid=0).

## Development
### Code Quality
```bash
uv run ruff check
uv run ruff format
```
