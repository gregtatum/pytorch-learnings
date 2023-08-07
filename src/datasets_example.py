#!/usr/bin/env python3

"""
This file demonstrates how to load in ParaCrawl through huggingface's "dataset" library.

https://huggingface.co/docs/datasets/index
https://huggingface.co/datasets/para_crawl/viewer/enes/train
"""

from datasets import load_dataset_builder, load_dataset

print("Looking up the dataset")
# Load the ParaCrawl dataset
ds_builder = load_dataset_builder("para_crawl", "enes")
print(f" > {ds_builder.info.description}")
print(f" > {ds_builder.info.homepage}")

# print(ds_builder.info)
#
# >>> DatasetInfo(
# >>>     description="Web-Scale Parallel Corpora for Official European Languages.",
# >>>     citation="@misc {paracrawl,\n    title  = {ParaCrawl},\n    year   = {2018},\n    url    = {http://paracrawl.eu/download.html.}\n}\n",
# >>>     homepage="https://paracrawl.eu/releases.html",
# >>>     license="",
# >>>     features={"translation": Translation(languages=("en", "es"), id=None)},
# >>>     post_processed=None,
# >>>     supervised_keys=SupervisedKeysData(input="en", output="es"),
# >>>     task_templates=None,
# >>>     builder_name="para_crawl",
# >>>     dataset_name="para_crawl",
# >>>     config_name="enes",
# >>>     version="1.0.0",
# >>>     splits={
# >>>         "train": SplitInfo(
# >>>             name="train",
# >>>             num_bytes=6209466040,
# >>>             num_examples=21987267,
# >>>             shard_lengths=None,
# >>>             dataset_name=None,
# >>>         )
# >>>     },
# >>>     download_checksums=None,
# >>>     download_size=1953839527,
# >>>     post_processing_size=None,
# >>>     dataset_size=6209466040,
# >>>     size_in_bytes=None,
# >>> )

print("Loading data set")
dataset = load_dataset("para_crawl", "enes", split="train")

print("Splitting data")
dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=1234)

print("\nShowing example data")
for row in dataset["test"]["translation"][0:5]:
    en = row["en"]
    es = row["es"]
    print(f"en: {en}")
    print(f"es: {es}")
