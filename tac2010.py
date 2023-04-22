# Running experiments in TAC 2010 

import sys, os, pickle
import pandas

import evalbase
sys.path.append(os.path.join(evalbase.path, "SueNes/human/tac"))
import tac
from eval_utils import eval_and_write



# 43 machine + 4 human summarizers per docsetID
# Each docsetID has 10 articles 

def clean_text(s: str):
    """Clean up the text in doc or summ in RealSumm dataset
    including, removing HTML tags, unescap HTML control sequences 
    """
    s = s.replace("<t>", "")
    s = s.replace("</t>", "")
    s = s.replace("\t", " ")
    s = s.strip()
    return s


def merge_article_summary_score(articles, summaries, scores, debug=False):
    columns = [
        "docsetID",
        "System",
        # we need docsetID so later we can use groupby easily to average across multiple documents corresponds to one summary
        "ArticleText", "ReferenceSummary", "SystemSummary",
        "Pyramid", "Linguistic", "Overall"  # the human scores
    ]

    counter = 0
    dataset_df = pandas.DataFrame(columns=columns)
    for docID, summary_dict in summaries.items():
        for articleID in range(10):
            ArticleText = " ".join(articles[docID][articleID])
            # sentences were cut into lists of strings in TAC
            for System, summary_sentences in summary_dict.items():
                row = {
                    "docsetID": [docID],
                    "ArticleText": [ArticleText],
                    "System": [System],
                    "ReferenceSummary": ["Place Holder"],  # TODO where is TAC's reference summary?
                    "SystemSummary": [" ".join(summary_sentences)],
                    "Pyramid": [scores[docID][System][0]],
                    "Linguistic": [scores[docID][System][1]],
                    "Overall": [scores[docID][System][2]]
                }

                tmp_dataset_df = pandas.DataFrame.from_dict(row)

                dataset_df = pandas.concat([dataset_df, tmp_dataset_df], ignore_index=True)
        counter += 1
        if debug and counter > 3:
            break

    
    # Set Pyramind to float
    dataset_df["Pyramid"] = dataset_df["Pyramid"].astype(float)

    # Set Linguistic and Overall to int
    dataset_df["Linguistic"] = dataset_df["Linguistic"].astype(int)
    dataset_df["Overall"] = dataset_df["Overall"].astype(int)

    return dataset_df


def load_tac(dataroot: str, debug=False):
    """

    We assume that you have fully recursively extracted the two files. 
    - [`GuidedSumm2010_eval.tgz`](https://tac.nist.gov/protected/past-aquaint-aquaint2/2010/GuidedSumm2010_eval.tgz) Downloadable from web, containing human evaluation results and system summaries. 
    - `TAC2010_Summarization_Documents.tgz` Emailed by NIST, containing the documents for which summaries are generated and rated. 
    Both files require you to apply to NIST for access. 

    The _dataroot_ directory should have the following structure: 
    dataroot 
    ├── GuidedSumm2010_eval
    │   ├── BE
    │   │   ├── models
    │   │   └── peers
    │   ├── manual
    │   │   ├── models
    │   │   ├── peers
    │   │   └── pyramids
    │   └── ROUGE
    │       ├── models
    │       └── peers
    └── TAC2010_Summarization_Documents
        └── GuidedSumm10_test_docs_files
            ├── D1001A
            │   ├── D1001A-A
            │   └── D1001A-B
            ├── D1002A
            │   ├── D1002A-A
            │   └── D1002A-B
            ├── D1003A
            │   ├── D1003A-A
            │   └── D1003A-B
            ├── D1004A
            │   ├── D1004A-A
            │   └── D1004A-B
            ... abridged ... 

    """
    article_set_path = os.path.join(dataroot, "TAC2010_Summarization_Documents/GuidedSumm10_test_docs_files/")
    summary_set_path = os.path.join(dataroot, "GuidedSumm2010_eval/ROUGE")
    human_score_path = os.path.join(dataroot, "GuidedSumm2010_eval/manual")

    # rouge_score_path = os.path.join(dataroot, "GuidedSumm2010_eval/ROUGE/rouge_A.m.out")

    setIDs = ["A"]  # we only use set A because set B is not applicable 
    sentence_delimiter = "  "
    summary_types = ["peers", "models"]

    articles = tac.get_articles(article_set_path, setIDs, sentence_delimiter)
    # _,_,_ = get_statistics(articles)

    summaries = tac.get_summaries(summary_set_path, setIDs, sentence_delimiter, summary_types)
    # sentence_delimiter,  NOT IN USE

    scores = tac.get_scores(human_score_path, summary_types, setIDs)

    dataset_df = merge_article_summary_score(articles, summaries, scores, debug=debug)

    return dataset_df


def main(exp_config: dict):
    # FIXME: This is cyclic import 
    tac_df_path = os.path.join(evalbase.path, "dataloader/tac_df.pkl")

    if os.path.exists(tac_df_path):
        dataset_df = pickle.load(open(tac_df_path, "rb"))
    else:
        dataset_df = load_tac(exp_config["data_path"], debug=False)
        pickle.dump(dataset_df, open(tac_df_path, "wb"))

    dataset_name = exp_config["dataset_name"]

    eval_and_write(dataset_df, dataset_name, exp_config)
