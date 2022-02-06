import os

from get_data import *
from preprocess_data import *
from baseline_model import *
from model_pytorch import *


def main():
    pytorch = False
    os.chdir(os.path.dirname(os.path.abspath("SB_main.py")))
    if pytorch:
        practice()
    else:
        input_dir = "data/"
        df = get_individual_data_files(input_dir, get_final=False)
        df = preprocess_dataframe(df)
        df.to_csv(input_dir+"final_data.csv", index=False)
        base_model(df)


if __name__ == '__main__':
    main()
