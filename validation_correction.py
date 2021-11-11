import numpy as np
import random
import pandas as pd

def post_prediction_correction(X_val, y_val, relationship_model):
    bs_sample = []
    # Obtain a 2d list of our covariate and y from validation pairs
    for i in range(len(X_val)):
        bs_sample.append(X_val[i] + [y_val[i]])

    return bootstrap_correction(bs_sample, relationship_model)


def bootstrap_correction(bs_sample, relationship_model, B=50000):
    list_of_beta_b = []
    # bootstrap iterations
    for b in range(B):
        print("Working on bootstrap iteration #: " + str(b), end='\r')
        # randomly sample with replacement from bootstrap sample
        cov_y_b = random.choices(bs_sample, k=14)
        # extract the y_predicted from bootstrap sample
        y_b = [[1, i[-1]] for i in cov_y_b]
        # correct the y_predicted using relationship model
        y_sim_b = np.array(relationship_model.predict(y_b))
        # extract all covariates from bootstrap sample
        cov_b = np.array([i[:-1] for i in cov_y_b])
        # calculate beta using formula (X_T_X)^-1 * X^-1 * y_sim_b
        beta_b = np.matmul(np.matmul(np.linalg.pinv(np.matmul(cov_b.T, cov_b)), cov_b.T), y_sim_b)
        # keep track of a list of betas
        list_of_beta_b.append(beta_b)

    df_beta_b = pd.DataFrame(list_of_beta_b)
    # take median of all betas and return
    df_beta_b = df_beta_b.apply(np.median, axis=0)
    return df_beta_b.values


def correct_validation(beta_bs, X_val):
    return np.matmul(X_val, beta_bs)
