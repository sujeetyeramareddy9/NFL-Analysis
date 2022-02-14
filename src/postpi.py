import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random


def figure3_plots(y_pred, y_test):
    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(6, 6)

    best_fit = np.arange(min(y_test), max(y_test))

    axes.scatter(x=y_test, y=y_pred, c="purple", alpha=0.35)
    axes.plot(best_fit, best_fit, c="red")
    axes.set(xlabel="y-observed", ylabel="y-predicted")
    axes.set_title("Sci-Kit Learn MLPRegressor")

    plt.savefig("src/plots/postpi_Fig2.png")


def bootstrap_(x_val, y_val_pred, y_val_actual, relationship_model, param=True, B=100):
    bootstrap_sample_pairs = []

    for i in range(len(x_val)):
        bootstrap_sample_pairs.append([x_val.values[i]] + [y_val_pred[i]])

    beta_estimators = []
    se_beta_estimators = []
    
    for b in range(B):
        # sample from the validation set with replacement
        bs_sample = random.choices(bootstrap_sample_pairs, k=len(bootstrap_sample_pairs))
        
        covariates = np.array([i[0] for i in bs_sample])
        y_p_b = np.array([i[-1] for i in bs_sample])
        
        # correct predictions according to relationship model
        y_b = relationship_model.predict(sm.add_constant(y_p_b.reshape(-1,1)))
        
        # inference model - OLS
        inf_model = sm.OLS(y_b, sm.add_constant(covariates.reshape(-1,1))).fit()

        beta_estimators.append(inf_model.params[1])
        se_beta_estimators.append(inf_model.bse[1])
        
    beta_hat_boot = np.median(beta_estimators)
    se_hat_boot = None
    if param:
        se_hat_boot = np.median(se_beta_estimators)
    else:
        se_hat_boot = np.std(beta_estimators)
    
    return beta_hat_boot, se_hat_boot


def postprediction_inference():
    return