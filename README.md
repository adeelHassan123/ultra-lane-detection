# Lane Detection

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .
```

### Preprocess data (one-time)
```bash
python scripts/preprocess_data.py --src /path/to/raw --dst /path/to/processed
```

### Run one experiment
```bash
python scripts/run_experiment.py --exp e7
```

### Generate analysis figures
```bash
python scripts/run_analysis.py
```