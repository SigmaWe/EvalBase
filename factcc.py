from eval_utils import eval_and_write
import os
import pandas
import json


def process_factcc_line(data, i):
    fact_label = float(data["label"] == "CORRECT")
    sample = {"doc": data["text"], "sum": data["claim"],
              "human": fact_label, "id": i, "id0": "id0"}
    return sample


def load_factcc(filepath):
    lines = []
    with open(filepath, "r", encoding="UTF-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            processed_data = process_factcc_line(data, i)
            lines.append(processed_data)
    df = pandas.DataFrame(lines)
    return df


def main(exp_config: dict):
    FACTCC_PATH = exp_config["data_path"]
    FACTCC_SPLIT = exp_config["split"]
    # TODO load the other splits.
    filepath = os.path.join(FACTCC_PATH, FACTCC_SPLIT["test"])
    df = load_factcc(filepath)
    eval_and_write("factcc", df, {**exp_config, "eval_levels": ["system"]})
    eval_and_write("factcc-pooled", df, {**exp_config, "docID_column": "id0", "eval_levels": ["summary"]})
