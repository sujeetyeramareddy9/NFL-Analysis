import os
import sys
from get_data import *
from preprocess_data import *
from baseline_model import *
from model_pytorch import *

try:
    test_file = sys.argv[1]
except Exception as e:
    test_file = None

def main():
    # ensure that the working directory is the location of run.py script
    os.chdir(os.path.dirname(os.path.abspath("run.py")))
    # input directory for location of data relative to run.py
    input_dir = "data/"
    # bool variable to determine if we want to retrieve the final preprocessed dataframe
    get_final=True
    if not get_final:
        # 
        df = get_individual_data_files(input_dir, get_final=get_final)
        df = preprocess_dataframe(df)
        df.to_csv(input_dir+"final_data.csv", index=False)
    else:
        train = pd.read_csv("final_data/train.csv")
        if test_file is not None:
            test = pd.read_csv("final_data/test.csv")
        else:
            test = pd.read_csv("final_data/test.csv")
        mdl = build_model(train, test)


if __name__ == '__main__':
    main()
