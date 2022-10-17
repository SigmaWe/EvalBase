import sys
sys.path.append("/Users/turx/Projects/Research/dar/DocAsRef/")
import bertscore_sentence.eval as bertscore_sentence

models = ["rouge", "bleurt", "bertscore", "bertscore-sentence"]
datasets = {
    "newsroom": {
        "human_metrics": ["InformativenessRating", "RelevanceRating", "CoherenceRating", "FluencyRating"],
        "docID_column": "ArticleID",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary",
        "approaches": ["trad", "new"],
        "human_eval_only_path": "dataloader/newsroom-human-eval.csv",
        "human_eval_w_refs_path": "dataloader/newsroom_human_eval_with_refs.csv",
        "refs_path": "/media/forrest/12T_EasyStore1/data/NLP/resources/newsroom/test.jsonl"
    },
    "realsumm_abs": {
        "docID_column": "doc_id",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary",
        "human_metrics": ["litepyramid_recall"],
        "approaches": ["trad", "new"],
        "path": "dataloader/abs.pkl"
    },
    "realsumm_ext": {
        "docID_column": "doc_id",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary",
        "human_metrics": ["litepyramid_recall"],
        "approaches": ["trad", "new"],
        "data_path": "dataloader/ext.pkl"
    },
    "summeval": {
        "human_metrics": ["consistency", "relevance", "coherence", "fluency"],
        "docID_column": "id",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary_0",  # the id ranges from 0 to 10
        "approaches": ["trad", "new"],
        "data_path": "dataloader/summeval_annotations.aligned.paired.scored.jsonl"
    },
    "tac2010": {
        "human_metrics": ["Pyramid", "Linguistic", "Overall"],
        "approaches": ["new"],
        "docID_column": "docsetID"
    }
}
eval_metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum", "bertscore", "bleurt", "bertscore-sentence"]
corr_metrics = ["pearsonr", "kendalltau", "spearmanr"]
