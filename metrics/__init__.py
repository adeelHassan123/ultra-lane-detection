class MetricTracker:
    def __init__(self):
        self.values = {}

    def update(self, key: str, value: float):
        self.values.setdefault(key, []).append(float(value))

    def mean(self, key: str) -> float:
        vals = self.values.get(key, [])
        return sum(vals) / max(len(vals), 1)

    def summary(self) -> dict:
        return {key: self.mean(key) for key in self.values}
