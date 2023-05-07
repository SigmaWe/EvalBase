from eval_utils import eval_and_write
import os
import pandas
import json
from tqdm import tqdm


def process_frank_line(line):
    _doc = line["article"]
    _sum = line["summary"]
    sample = {"doc": _doc, "sum": _sum, "ref": line["reference"],
              "human": line["Factuality"], "id": line["hash"],  "id0": "id0"}

    return sample


def load_frank(data_file, annotation_file):
    """Return [cnndm,xsum] variants dataframes.
    """
    with open(data_file, "r", encoding="UTF-8") as f:
        data = json.load(f)
    with open(annotation_file, "r", encoding="UTF-8") as f:
        annot = json.load(f)

    data_dict = {}

    for sample in data:
        data_dict[(sample['hash'], sample['model_name'])] = sample
    for sample in annot:
        data_dict[(sample['hash'], sample['model_name'])].update(sample)

    cnndm = []
    xsum = []

    for line in tqdm(data_dict.values()):
        processed_line = process_frank_line(line)
        if line["dataset"] == "cnndm":
            cnndm.append(processed_line)
        elif line["dataset"] == "bbc":
            xsum.append(processed_line)

    return [pandas.DataFrame(cnndm), pandas.DataFrame(xsum)]


def main(exp_config: dict):
    FRANK_PATH = exp_config["data_path"]

    cnndm, xsum = load_frank(os.path.join(FRANK_PATH, "benchmark_data.json"),
                             os.path.join(FRANK_PATH, "human_annotations.json"))
    eval_and_write("frank-cnndm", cnndm, exp_config)
    eval_and_write("frank-xsum", xsum, exp_config)
