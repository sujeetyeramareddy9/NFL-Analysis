import os
import sys
sys.path.insert(0, './src')
from get_data import *
from preprocess_data import *
from baseline_model import *
from model_MLP import *
from postpi import *

# adding try, except block so that we can use the test file as neccessary
try:
    test_file = sys.argv[1]
except Exception as e:
    test_file = None

def main(targets):
    if "test" in targets:
        # test data
        test = pd.read_csv("./src/test/test.csv")
    else:
        # train data
        train = pd.read_csv("./data/final_data/train.csv")


    # ensure that the working directory is the location of run.py script
    os.chdir(os.path.dirname(os.path.abspath("run.py")))
    # input directory for location of data relative to run.py
    input_dir = "src/data/"
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
    
    test = pd.read_csv("./src/test/test.csv")
    train = pd.read_csv("./src/final_data/train.csv")

    nn = True
    if nn:
        X_train, X_test, y_train, y_test, cn = get_data_ready_for_nn(train, test)
        mdl = train_nn(X_train, X_test, y_train, y_test, cn)
    else:
        mdl = build_model(train, test)

    y_test_baseline, y_pred_baseline = build_model(train, test)

    superbowl_pred = pd.read_csv("./src/final_data/superbowls.csv").to_numpy()
    superbowl_pred = mdl.predict(superbowl_pred)
    print(superbowl_pred)

    # Post-prediction inference
    #postprediction_inference(X_test, y_test, prediction_mdl, y_test_baseline, y_pred_baseline)


if __name__ == '__main__':
    targets =  sys.argv[1:]
    main(targets)
 
