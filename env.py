import functools
import os

evalbase_path = os.path.dirname(os.path.abspath(__file__))

# fix: GPU OOM (TF exhausts GPU memory, crashing PyTorch)
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

datasets = {
    "newsroom": {
        "human_metrics": ["InformativenessRating", "RelevanceRating", "CoherenceRating", "FluencyRating"],
        "docID_column": "ArticleID",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary",
        "approaches": ["trad", "new"],
        "human_eval_only_path": os.path.join(evalbase_path, "dataloader/newsroom-human-eval.csv"),  # you need to get this file. See ReadMe.
        "refs_path": os.path.join(evalbase_path, "dataloader/test.jsonl"),  # you need to get this file. See ReadMe.
        "human_eval_w_refs_path": os.path.join(evalbase_path, "dataloader/newsroom_human_eval_with_refs.csv")
    },
    "realsumm_abs": {
        "docID_column": "doc_id",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary",
        "human_metrics": ["litepyramid_recall"],
        "approaches": ["trad", "new"],
        "data_path": os.path.join(evalbase_path, "dataloader/abs.pkl")  # you need to get this file. See ReadMe.
    },
    "realsumm_ext": {
        "docID_column": "doc_id",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary",
        "human_metrics": ["litepyramid_recall"],
        "approaches": ["trad", "new"],
        "data_path": os.path.join(evalbase_path, "dataloader/ext.pkl")  # you need to get this file. See ReadMe.
    },
    "summeval": {
        "human_metrics": ["consistency", "relevance", "coherence", "fluency"],
        "docID_column": "id",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary_0",  # the id ranges from 0 to 10
        "approaches": ["trad", "new"],
        "data_path": os.path.join(evalbase_path, "dataloader/summeval_annotations.aligned.paired.scored.jsonl")
    },
    "tac2010": {
        "human_metrics": ["Pyramid", "Linguistic", "Overall"],
        "approaches": ["new"],
        "docID_column": "docsetID"
    }
}

import spacy
nlp_spacy = spacy.load("en_core_web_lg")

metrics = {}  # your metrics here

corr_metrics = ["pearsonr", "kendalltau", "spearmanr"]
