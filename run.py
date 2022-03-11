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
    # ensure that the working directory is the location of run.py script
    os.chdir(os.path.dirname(os.path.abspath("run.py")))

    # bool variable to determine if we want to retrieve the final preprocessed dataframe
    get_final=True
    if not get_final:
        # in this case, we want to run through the ETL for the data

        # extract
        df = get_individual_data_files("./src/data/", get_final=get_final)

        # transform
        df = preprocess_dataframe(df)

        # load
        df.to_csv("./src/data/final_data.csv", index=False)
    else:
        df = pd.read_csv("./src/data/final_data.csv")

    post_pi = True

    if "test" in targets:
        # test data
        train = pd.read_csv("./src/final_data/train.csv")[:10]
        test = pd.read_csv("./src/test/test.csv")[10:15]
        post_pi = False
    else:
        # train data
        train = pd.read_csv("./src/final_data/train.csv")
        test = pd.read_csv("./src/final_data/test.csv")
        
    
    y_test_baseline, y_pred_baseline = build_model(train, test)

    X_train, X_test, y_train, y_test, cn = get_data_ready_for_nn(train, test)
    prediction_mdl = train_nn(X_train, X_test, y_train, y_test, cn)

    superbowl_pred = pd.read_csv("./src/final_data/superbowl.csv").to_numpy()
    superbowl_pred = prediction_mdl.predict(superbowl_pred)
    print(f"Superbowl Prediction: Rams will win by {np.round(superbowl_pred[0], 0)} points.")


    if post_pi:
        # Post-prediction inference
        postprediction_inference(X_test, y_test, prediction_mdl, y_test_baseline, y_pred_baseline)


if __name__ == '__main__':
    targets =  sys.argv[1:]
    main(targets)
 
