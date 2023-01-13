# Print the JSON output into a CSV format for importing into Excel/Google Sheets

#%%
import json
import pandas as pd
from typing import List, Optional, Union

#%% Load results JSON using Pandas
def load_result_json_pandas(json_result: str):
    df = pd.read_json(json_result)
    return df

#%% transform result dataframe

def transform_dataframe(df, reset_index:bool) -> pd.Series:

    # Convert keys in JSON to multiindexes 
    # This is probably due to that we stupidly set orient='index' 
    # when dumping result DataFrame to JSON 
    df.columns = pd.MultiIndex.from_tuples(
        [eval(column) for column in df.columns], 
        # because loaded JSON has strings of tuple definitions
        names = ( # multiindex names 
            "corr_metric", 
            "aspect", 
            "approach", 
            "model_name", 
            "scorer_name" 
            )
        )

    # Transpose indexes and columns 
    df = df.T 

    # Leave only the average scores
    s = df["average"]
    # This resets to pandas.Series, with multiindex as index

    if reset_index:
        # This will insert repeating column names. 
        # Default, False 
        s.reset_index()

    return s 

# the result JSON, once loaded, has the following structure:
# keys are tuples, (corr_metric, aspect, trad/new, metric_name, metric_)

# %% beautiful print 

def beautiful_print(
    s: pd.Series, 
    corr_metric: str,
    aspects: List[str],
    approaches: List[str],
    model_scorer_tuples: List[tuple[str, str]] | None, 
    ):
    """
    aspect: List[str], dependending on the dataset
    corr_metric: str, a correlation metric, 
        e.g., "spearmanr", "pearsonr", or "kendalltau"
    approach: List[str], e.g., ["trad", "new"]
    model_score_tuples: list of tuples of (model_name, score_name), e.g., 
       [("bertscore", "f1"), ("PreCalc", "supert")]
       When None, print all models and scores
    
    """

    # Step 1: build the table content 
    print_table = [] # 2D list of strings/floats

    # narrow to the correlation metric
    s = s[s.index.get_level_values('corr_metric') == corr_metric]

    for aspect in aspects:
        for approach in approaches:

            if model_scorer_tuples is None:
                # narrow to the aspect
                s1 = s[s.index.get_level_values('aspect') == aspect]
                # narrow to the approach
                s1 = s1[s1.index.get_level_values('approach') == approach]

                column = s1.values.round(decimals=3).tolist()
            else: # get values for individual pairs of (model, scorer)
                column = []
                for model, scorer in model_scorer_tuples:
                    if model == "PreCalc" :
                        continue 
                    value = s.loc[(corr_metric, aspect, approach, model, scorer)]
                    column.append(f"{value:.3f}")

            print_table.append(column)
    

    # Step 2: build the table header

    approach_mapping = {"trad":"Ref-Based", "new":"Ref-Free"}

    # first column is empty, for model x scorer
    header1 = [" "]
    for aspect in aspects:
        header1 += [aspect]*len(approaches)
    header2 = [" "] + \
        [approach_mapping[approach] for approach in approaches]*len(aspects)
    # header1 = "\t".join(header1)
    # header2 = "\t".join(header2)

    # Step 3: print the table
    if model_scorer_tuples is None:
        model_names = s.index.get_level_values('model_name').to_list()
        scorer_names = s.index.get_level_values('scorer_name').to_list()
        model_scorer_tuples = zip(model_names, scorer_names)

    first_column = [
        f"{model}_{scorer}" 
        for model, scorer in model_scorer_tuples
        ]

    print_table = [first_column] + print_table

    Rows = list(zip(*print_table))
    Rows = [header1 , header2] + Rows

    # Code taken from https://stackoverflow.com/questions/13214809/pretty-print-2d-list
    s = [[str(e) for e in row] for row in Rows]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print ('\n'.join(table))


# %%
if __name__ == "__main__":

    # Configs 
    pandas_json_path = "summeval_summary.json"
    aspects = ["consistency", "relevance", "coherence", "fluency"]
    corr_metrics = ["spearmanr", "pearsonr", "kendalltau"]
    approaches = ["new", "trad"]
    model_scorer_tuples = [("bertscore", "f1"), ("PreCalc", "bleurt")]
    # model_scorer_tuples = None 
    # End of configs

    df = load_result_json_pandas(pandas_json_path)
    s  = transform_dataframe(df, reset_index=False)

    for corr_metric in corr_metrics:
        print ("\n"+f"Correlation metric: {corr_metric}" + "="*10)
        beautiful_print(
            s, 
            corr_metric=corr_metric, 
            aspects=aspects , 
            approaches=approaches, 
            model_scorer_tuples=model_scorer_tuples
            )
# %%
