import os
import sys
from get_data import *
from preprocess_data import *
from baseline_model import *
from model_pytorch import *

# adding try, except block so that we can use the test file as neccessary
try:
    test_file = sys.argv[1]
except Exception as e:
    test_file = None

def main():
    if "test" in targets:
        # test data
        test = pd.read_csv("./test/test.csv")
    else:
        # train data
        train = pd.read_csv("./data/final_data/train.csv")


    # ensure that the working directory is the location of run.py script
    os.chdir(os.path.dirname(os.path.abspath("run.py")))
    # input directory for location of data relative to run.py
    input_dir = "data/"
    # bool variable to determine if we want to retrieve the final preprocessed dataframe
    get_final=True
    if not get_final:
        # in this case, we want to run through the ETL for the data

        # extract
        df = get_individual_data_files(input_dir, get_final=get_final)

        # transform
        df = preprocess_dataframe(df)

        # load
        df.to_csv(input_dir+"final_data.csv", index=False)
    else:
        # in this case, we read directly from exported final data


        # build our model
        mdl = build_model(train, test)

        # code here is similar for our neural network once "model_pytorch" is developed


if __name__ == '__main__':

    targets =  sys.arg[1:]
    main(targets)
