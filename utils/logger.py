import csv
import os

class CSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.header_written = False

    def log(self, data_dict):
        mode = "a" if os.path.exists(self.filepath) else "w"
        with open(self.filepath, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data_dict.keys())
            if mode == "w":
                writer.writeheader()
            writer.writerow(data_dict)

class ResultsLogger:
    def __init__(self, filepath):
        self.filepath = filepath

    def log(self, data_dict):
        mode = "a" if os.path.exists(self.filepath) else "w"
        with open(self.filepath, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data_dict.keys())
            if mode == "w":
                writer.writeheader()
            writer.writerow(data_dict)

