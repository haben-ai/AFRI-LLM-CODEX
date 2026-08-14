"""Sample module with an intentional memory leak, for testing run_inference.py."""

import os

_CACHE = []  # module-level cache that is never cleared -> grows forever


def load_record(path):
    with open(path, "rb") as f:
        return f.read()


def process_data(items):
    """Reads every item into memory and never releases it."""
    results = []
    for item in items:
        record = load_record(item)
        _CACHE.append(record)      # leak: appended but never removed
        results.append(len(record))
    return results


class DataPipeline:
    def __init__(self, directory):
        self.directory = directory

    def run(self):
        items = [os.path.join(self.directory, f) for f in os.listdir(self.directory)]
        return process_data(items)
