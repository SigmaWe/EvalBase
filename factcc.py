from eval_utils import eval_and_write
import numpy as np
import os
import pandas
import json
from tqdm import tqdm

import json

import pandas


# factCC

def process_factcc_line(data, i):
    fact_label = float(data["label"] == "CORRECT")
    sample = {"doc": data["text"], "sum": data["claim"],
              "human": fact_label, "id": i, "id0": "id0"}
    return sample


def load_factcc(filepath, limit=None):
    lines = []
    with open(filepath, "r", encoding="UTF-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            processed_data = process_factcc_line(data, i)
            lines.append(processed_data)
    df = pandas.DataFrame(lines)
    return df


# Frank


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

# qags


def process_qags_line(line, i):
    _doc = line["article"]

    sum_texts = []
    facts = []
    for sent in line["summary_sentences"]:
        sum_texts.append(sent["sentence"])
        response = [worker["response"] ==
                    "yes" for worker in sent["responses"]]
        facts.extend(response)

    _sum = " ".join(sum_texts)
    sample = {"doc": _doc, "sum": _sum,
              "human": np.mean(facts), "id": i,  "id0": "id0"}

    return sample


def load_qags(filepath):
    lst = []
    with open(filepath, "r", encoding="UTF-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            row = process_qags_line(data, i)
            lst.append(row)
    df = pandas.DataFrame(lst)
    return df


####################
# The main functions

def factCC_main(exp_config: dict):
    FACTCC_PATH = exp_config["data_path"]
    FACTCC_SPLIT = exp_config["split"]
    # TODO load the other splits.
    filepath = os.path.join(FACTCC_PATH, FACTCC_SPLIT["test"])
    df = load_factcc(filepath)
    eval_and_write("factcc", df, {**exp_config, "eval_levels": ["system"]})
    eval_and_write("factcc-pooled", df, {**exp_config, "docID_column": "id0", "eval_levels": ["summary"]})


def frank_main(exp_config: dict):
    FRANK_PATH = exp_config["data_path"]

    cnndm, xsum = load_frank(os.path.join(FRANK_PATH, "benchmark_data.json"),
                             os.path.join(FRANK_PATH, "human_annotations.json"))
    eval_and_write("frank-cnndm", cnndm, exp_config)
    eval_and_write("frank-xsum", xsum, exp_config)


def qags_main(exp_config: dict):
    # TODO parameterize the path
    QGAS_PATH = exp_config["data_path"]
    cnndm = load_qags(os.path.join(QGAS_PATH, "mturk_cnndm.jsonl"))
    xsum = load_qags(os.path.join(QGAS_PATH, "mturk_xsum.jsonl"))
    # DEBUG using part of the data
    eval_and_write("qags-cnndm", cnndm, {**exp_config, "eval_levels": ["system"]})
    eval_and_write("qags-xsum", cnndm, {**exp_config, "docID_column": "id0", "eval_levels": ["system"]})
    eval_and_write("qaqs-cnndm-pooled", cnndm, {**exp_config, "docID_column": "id0", "eval_levels": ["summary"]})
    eval_and_write("qags-xsum-pooled", cnndm, {**exp_config, "docID_column": "id0", "eval_levels": ["summary"]})
