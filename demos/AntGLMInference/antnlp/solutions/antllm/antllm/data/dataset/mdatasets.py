try:
    from datasets import Dataset, load_dataset
except Exception:
    Dataset = None
    load_dataset = None

Dataset = Dataset
load_dataset = load_dataset
