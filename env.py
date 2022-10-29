import sys
import functools 
import evaluate

datasets = {
    "newsroom": {
        "human_metrics": ["InformativenessRating", "RelevanceRating", "CoherenceRating", "FluencyRating"],
        "docID_column": "ArticleID",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary",
        "approaches": ["trad", "new"],
        "human_eval_only_path": "dataloader/newsroom-human-eval.csv",  # you need to get this file. See ReadMe.
        "refs_path": "dataloader/test.jsonl",  # you need to get this file. See ReadMe.
        "human_eval_w_refs_path": "dataloader/newsroom_human_eval_with_refs.csv",
        "result_summary_simple_path": "results/newsroom_summary.txt",
        "result_summary_detail_path": "results/newsroom_summary.json",
        "result_system_simple_path": "results/newsroom_system.txt",
        "result_system_detail_path": "results/newsroom_system.json"
    },
    "realsumm_abs": {
        "docID_column": "doc_id",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary",
        "human_metrics": ["litepyramid_recall"],
        "approaches": ["trad", "new"],
        "data_path": "dataloader/abs.pkl",  # you need to get this file. See ReadMe.
        "result_summary_simple_path": "results/realsumm_abs_summary.txt",
        "result_summary_detail_path": "results/realsumm_abs_summary.json",
        "result_system_simple_path": "results/realsumm_abs_system.txt",
        "result_system_detail_path": "results/realsumm_abs_system.json"
    },
    "realsumm_ext": {
        "docID_column": "doc_id",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary",
        "human_metrics": ["litepyramid_recall"],
        "approaches": ["trad", "new"],
        "data_path": "dataloader/ext.pkl",  # you need to get this file. See ReadMe.
        "result_summary_simple_path": "results/realsumm_ext_summary.txt",
        "result_summary_detail_path": "results/realsumm_ext_summary.json",
        "result_system_simple_path": "results/realsumm_ext_system.txt",
        "result_system_detail_path": "results/realsumm_ext_system.json"
    },
    "summeval": {
        "human_metrics": ["consistency", "relevance", "coherence", "fluency"],
        "docID_column": "id",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary_0",  # the id ranges from 0 to 10
        "approaches": ["trad", "new"],
        "data_path": "dataloader/summeval_annotations.aligned.paired.scored.jsonl",
        "result_summary_simple_path": "results/summeval_summary.txt",
        "result_summary_detail_path": "results/summeval_summary.json",
        "result_system_simple_path": "results/summeval_system.txt",
        "result_system_detail_path": "results/summeval_system.json"
    },
    "tac2010": {
        "human_metrics": ["Pyramid", "Linguistic", "Overall"],
        "approaches": ["new"],
        "docID_column": "docsetID"
    }
}

metrics = {
    # "bleurt": evaluate.load('bleurt', config_name='BLEURT-20', module_type='metric').compute,
    "rouge":  functools.partial(evaluate.load("rouge").compute,  use_aggregator=False),
    "bertscore":  functools.partial(evaluate.load("bertscore").compute, lang='en', use_fast_tokenizer=True),
}


corr_metrics = ["pearsonr", "kendalltau", "spearmanr"]
