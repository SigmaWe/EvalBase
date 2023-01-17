import os
import warnings
import pandas
import sys
import datetime

path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(path, "results")


def create_result_path():
    if not os.path.exists(result_path):
        os.makedirs(result_path)

import factcc

def eval():
    create_result_path()
    print("==== env.factcc.qags_main(), size: 235")
    factcc.qags_main()
    print("==== env.factcc.frank_main(): size: 1250")
    factcc.frank_main()
    print("==== env.factcc.factCC_main(): size: large")
    factcc.factCC_main()
    

if __name__ == '__main__':
    eval()