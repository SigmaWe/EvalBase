import numpy as np
import os
import pandas
import json
from tqdm import tqdm

import json
import typing

import pandas

import env
import evalbase


# factCC

def process_factcc_line(data, i):
    fact_label = float(data["label"] == "CORRECT")
    sample = {"doc": data["text"], "sum": data["claim"], "human": fact_label, "id": i}
    return sample

def load_factcc(filepath, limit=None):
    lines = []
    with open(filepath, "r", encoding="UTF-8") as f:
        for i,line in enumerate(f):
            data = json.loads(line)
            processed_data = process_factcc_line(data, i)
            lines.append(processed_data)
    df = pandas.DataFrame(lines)
    return df


# Frank


def process_frank_line(line):
    _doc = line["article"]
    _sum = line["summary"]
    sample = {"doc": _doc, "sum": _sum, "ref": line["reference"], "human": line["Factuality"], "id": line["hash"]}

    return sample

def load_frank(data_file, annotation_file):
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
        if line["dataset"] == "cnndm": cnndm.append(processed_line)
        elif line["dataset"] == "bbc": xsum.append(processed_line)
    
    # use cnndm for now
    df = pandas.DataFrame(cnndm)
    return df

# qags

def process_qags_line(line, i):
    _doc = line["article"]
    
    sum_texts = []
    facts = []
    for sent in line["summary_sentences"]:
        sum_texts.append(sent["sentence"])
        response = [worker["response"] == "yes" for worker in sent["responses"]]
        facts.extend(response)
    
    _sum = " ".join(sum_texts)
    sample = {"doc": _doc, "sum": _sum, "human": np.mean(facts), "id": i}

    return sample


def load_qags(filepath):
    lst = []
    with open(filepath, "r", encoding="UTF-8") as f:
        for i,line in enumerate(f):
            data = json.loads(line)
            row = process_qags_line(data, i)
            lst.append(row)
    df = pandas.DataFrame(lst)
    return df

def do_eval(dataset_df, config):
    # using cnndm only
    import eval_utils

    print(f"{config['name']} Summary-Level")
    corr_df = eval_utils.eval_summary_level(
        dataset_name=config['name'],
        dataset_df=dataset_df[:10],
        exp_approaches=config["approaches"],
        exp_models=env.metrics,
        corr_metrics=env.corr_metrics,
        document_column=config["document_column"],
        docID_column=config["docID_column"],
        system_summary_column=config["system_summary_column"],
        reference_summary_column=config["reference_summary_column"],
        human_metrics=config["human_metrics"],
        # pre_calculated_metrics=precalc_metrics,
        debug=False
    )
    eval_utils.write_results(
        simple_df=corr_df['average'],
        detail_df=corr_df,
        simple_path=f"results/{config['name']}_summary.txt",
        detail_path=f"results/{config['name']}_summary.json"
    )

    print(f"{config['name']} System-Level")
    corr_df = eval_utils.eval_system_level(
        dataset_name=config['name'],
        dataset_df=dataset_df,
        exp_approaches=config["approaches"],
        exp_models=env.metrics,
        corr_metrics=env.corr_metrics,
        document_column=config["document_column"],
        docID_column=config["docID_column"],
        system_summary_column=config["system_summary_column"],
        reference_summary_column=config["reference_summary_column"],
        human_metrics=config["human_metrics"],
        # pre_calculated_metrics=precalc_metrics,
        debug=False
    )
    eval_utils.write_results(
        simple_df=corr_df,
        simple_path=f"results/{config['name']}_system.txt",
        detail_path=f"results/{config['name']}_system.json"
    )

    print("End")


# The main functions

def factCC_main():
    DATA_ROOT = "/home/hebi/git/factRel/data/dataset/"
    FACTCC_PATH = os.path.join(DATA_ROOT, "factCC/data_pairing/data/generated_data/data-clipped")

    FACTCC_SPLIT = {
        "train" : "data-train.jsonl",
        "dev" : "data-dev.jsonl",
        "test" : "data-test.jsonl"
    }
    config = {
        'human_metrics': ['human'],
        'docID_column': 'id',
        'document_column': 'doc',
        'system_summary_column': 'sum',
        # FIXME only one summary is available
        'reference_summary_column': 'sum',
        'approaches': ['new'],
    }
    # TODO load the other splits.
    filepath = os.path.join(FACTCC_PATH, FACTCC_SPLIT["test"])
    df = load_factcc(filepath)
    do_eval(df, config)

def frank_main():
    DATA_ROOT = "/home/hebi/git/factRel/data/dataset/"
    FRANK_FOLDER = os.path.join(DATA_ROOT, 'frank/data')
    FRANK_DATA = 'benchmark_data.json'
    FRANK_ANNOTATION = 'human_annotations.json'

    config = {
        'name': "frank",
        'human_metrics': ['human'],
        'docID_column': 'id',
        'document_column': 'doc',
        'system_summary_column': 'sum',
        'reference_summary_column': 'ref',
        'approaches': ['new'],
    }
    df = load_frank(os.path.join(FRANK_FOLDER, FRANK_DATA),
            os.path.join(FRANK_FOLDER, FRANK_ANNOTATION))
    do_eval(df, config)


def qags_main():
    # TODO parameterize the path
    DATA_ROOT = "/home/hebi/git/factRel/data/dataset/"
    QGAS_FOLDER = os.path.join(DATA_ROOT, "qags/data")
    QGAS_FILES = ["mturk_cnndm.jsonl", "mturk_xsum.jsonl"]
    config = {
        'name': "factcc",
        'human_metrics': ['human'],
        'docID_column': 'id',
        'document_column': 'doc',
        'system_summary_column': 'sum',
        # FIXME only one summary is available
        'reference_summary_column': 'sum',
        'approaches': ['new'],
    }
    dataset_df = load_qags(os.path.join(QGAS_FOLDER, "mturk_cnndm.jsonl"))
    do_eval(dataset_df, config)

