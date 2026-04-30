# `tests/` Guide

Unit tests verify core pipeline pieces before expensive training runs.

## Current test scope

- checkpoint persistence
- dataset output structure
- loss gradient propagation
- segmentation metric behavior
- model output shape

## Recommended usage

Run tests before every long experiment batch:

```bash
python -m pytest -v
```
