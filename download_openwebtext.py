import json
from datasets import load_from_disk, DatasetDict, load_dataset

#downloading the full datset locally
dataset = load_dataset("openwebtext")
dataset.save_to_disk("openwebtext_local")

# Loading the dataset and making splits from it
dataset = load_from_disk("openwebtext_local")

def save_split_to_json(dataset_split, filename):
    data = [{"text": example["text"]} for example in dataset_split]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

total_size = len(dataset["train"])
val_size = int(0.1 * total_size)
test_size = int(0.1 * total_size)

shuffled_dataset = dataset["train"].shuffle(seed=42)

validation_set = shuffled_dataset.select(range(val_size))
test_set = shuffled_dataset.select(range(val_size, val_size + test_size))
train_set = shuffled_dataset.select(range(val_size + test_size, total_size))

split_dataset = DatasetDict({
    "train": train_set,
    "validation": validation_set,
    "test": test_set
})

save_split_to_json(split_dataset["validation"], "openwebtext_val.json")
save_split_to_json(split_dataset["test"], "openwebtext_test.json")

