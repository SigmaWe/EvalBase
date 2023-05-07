from eval_utils import eval_and_write
import numpy as np
import os
import pandas
import json


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


def main(exp_config: dict):
    # TODO parameterize the path
    QGAS_PATH = exp_config["data_path"]
    cnndm = load_qags(os.path.join(QGAS_PATH, "mturk_cnndm.jsonl"))
    xsum = load_qags(os.path.join(QGAS_PATH, "mturk_xsum.jsonl"))
    # DEBUG using part of the data
    eval_and_write("qags-cnndm", cnndm, {**exp_config, "eval_levels": ["system"]})
    eval_and_write("qags-xsum", cnndm, {**exp_config, "docID_column": "id0", "eval_levels": ["system"]})
    eval_and_write("qaqs-cnndm-pooled", cnndm, {**exp_config, "docID_column": "id0", "eval_levels": ["summary"]})
    eval_and_write("qags-xsum-pooled", cnndm, {**exp_config, "docID_column": "id0", "eval_levels": ["summary"]})
