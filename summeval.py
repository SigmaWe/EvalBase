import json, typing, os
import pandas
import eval_utils
import evaluate


def clean_text(s: str):
    s = s.replace("\t", " ")
    s = s.strip()
    return s


def pool_human_rating(
        human_ratings: typing.List[dict],
        pool_method: str = "mean") \
        -> dict:
    # input list:
    # [{'coherence': 2, 'consistency': 1, 'fluency': 4, 'relevance': 2},
    #  {'coherence': 1, 'consistency': 1, 'fluency': 2, 'relevance': 1},
    #  {'coherence': 1, 'consistency': 1, 'fluency': 3, 'relevance': 2}]

    df = pandas.DataFrame(human_ratings)

    if pool_method == "mean":
        q = df.mean()

    return q.to_dict()

    # ratings = {}
    # for human_metric in ['coherence', 'consistency', 'fluency', 'relevance']: 
    #     tmp = 0 
    #     for i in range(3): 
    #         tmp += human_ratings[i][human_metric]

    #     if pool_method == "mean": 
    #         ratings[human_metric] = tmp/3

    # return ratings 


def load_summeval(paired_jsonl):
    human_metrics = ['coherence', 'consistency', 'fluency', 'relevance']
    with open(paired_jsonl, 'r', encoding='utf-8') as fd:
        dataset = [json.loads(line) for line in fd]

        df = pandas.DataFrame(dataset)

        # return df 
        # df.columns ==>
        # ['id', 'decoded', 'expert_annotations', 'turker_annotations',
        #    'references', 'model_id', 'filepath', 'metric_scores_1',
        #    'metric_scores_6', 'metric_scores_11', 'text']

        # process nested precalcualted metrics
        tdf = df['metric_scores_1'].to_list()
        for row in tdf:
            row.update(row['rouge'])
            row['supert'] = row['supert'][0]
            del row['rouge']
        df = pandas.concat([df, pandas.DataFrame(tdf)], axis=1)

        for refId in range(11):
            df[f"ReferenceSummary_{refId}"] = df["text"]  # place holder
        for human_metric in human_metrics:
            df[human_metric] = df["id"]  # place holder

        # clean up 
        df = df.rename(columns={'decoded': 'SystemSummary', 'text': 'ArticleText', 'model_id': 'system'})

        for index, row in df.iterrows():
            for refId in range(11):
                df.at[index, f"ReferenceSummary_{refId}"] = clean_text(row["references"][refId])

            pooled_human_ratings = pool_human_rating(row['expert_annotations'])
            for human_metric in human_metrics:
                df.at[index, human_metric] = pooled_human_ratings[human_metric]

            for column in ['ArticleText', 'SystemSummary']:
                df.at[index, column] = clean_text(row[column])

        df = df.drop(
            columns=['filepath', 'metric_scores_1', 'metric_scores_6', 'metric_scores_11', 'expert_annotations',
                     'turker_annotations', 'references'])

    return df

    # In [7]: df.iloc[1]["expert_annotations"]
    # Out[7]: 
    # [{'coherence': 3, 'consistency': 5, 'fluency': 5, 'relevance': 2},
    #  {'coherence': 2, 'consistency': 5, 'fluency': 5, 'relevance': 3},
    #  {'coherence': 2, 'consistency': 5, 'fluency': 5, 'relevance': 3}]


def main(exp_config: dict):
    dataset_name = exp_config["dataset_name"]
    dataset_df = load_summeval(exp_config["data_path"])
    eval_utils.eval_and_write(dataset_name, dataset_df, exp_config)


if __name__ == "__main__":
    import os, functools

    exp_config = {
        # about the dataset and dataframe 
        "dataset_name": "summeval",
        "human_metrics": ["consistency", "relevance", "coherence", "fluency"],
        "docID_column": "id",
        "document_column": "ArticleText",
        "system_summary_column": "SystemSummary",
        "reference_summary_column": "ReferenceSummary_0",  # the id ranges from 0 to 10
        # about the experiments 
        "nlg_metrics" : {
            "bleurt": evaluate.load('bleurt', config_name='BLEURT-20', module_type='metric').compute,
            "rouge":  functools.partial(evaluate.load("rouge").compute,  use_aggregator=False),
            "bertscore":  functools.partial(evaluate.load("bertscore").compute, lang='en', use_fast_tokenizer=True),
        }, 
        "corr_metrics" : ["spearman", "pearson", "kendalltau"], 
        "approaches": ["trad", "new"],
        "eval_levels": ["summary", "system"],
        "data_path": os.path.join(path, "dataloader/summeval_annotations.aligned.paired.scored.jsonl"),
        "result_path_root": "./results/",
        "precal_metrics": [  # keys from original SummEval json file
            'rouge_1_precision', 'rouge_1_recall', 'rouge_1_f_score',
            'rouge_2_precision', 'rouge_2_recall', 'rouge_2_f_score',
            'rouge_l_precision', 'rouge_l_recall', 'rouge_l_f_score',
            'rouge_we_1_p', 'rouge_we_1_r', 'rouge_we_1_f',
            'rouge_we_2_p', 'rouge_we_2_r', 'rouge_we_2_f',
            'meteor', 'cider', 's3_pyr', 's3_resp',
            'mover_score', 'sentence_movers_glove_sms', 'bleu',
            'bert_score_precision', 'bert_score_recall', 'bert_score_f1',
            'blanc', 'summaqa_avg_prob', 'summaqa_avg_fscore', 'supert']
    }

    main(exp_config)
