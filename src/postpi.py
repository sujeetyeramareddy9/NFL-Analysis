import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random
import pandas as pd
from sklearn.linear_model import LinearRegression


def figure2_plot(qb_rating, y_test, y_pred):
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)

    m, b = np.polyfit(qb_rating, y_test, 1)
    axes[0].scatter(x=qb_rating, y=y_test, c="blue", alpha=0.35)
    axes[0].plot(qb_rating, m*qb_rating + b, c="red")
    axes[0].set(xlabel="QB_Rating x", ylabel="Observed outcomes y")

    m, b = np.polyfit(qb_rating, y_pred, 1)
    axes[1].scatter(x=qb_rating, y=y_pred, c="red", alpha=0.35)
    axes[1].plot(qb_rating, m*qb_rating + b, c="red")
    axes[1].set(xlabel="QB_Rating x", ylabel="Predicted outcomes y")

    plt.savefig("src/plots/postpi_Fig2.png")
    plt.clf()


def figure3_plot(y_pred, y_test):
    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(6, 6)
    
    m, b = np.polyfit(y_test, y_pred, 1)
    axes.scatter(x=y_test, y=y_pred, c="purple", alpha=0.35)
    axes.plot(y_test, m*y_test + b, c="red")
    axes.set(xlabel="y-observed", ylabel="y-predicted")
    axes.set_title("MLPRegressor")

    plt.savefig("src/plots/postpi_Fig3.png")
    plt.clf()


def figure4_plot(all_true_outcomes, all_true_se, all_true_t_stats, all_nocorrection_estimates, all_parametric_estimates, all_nonparametric_estimates, all_nocorrection_se, all_parametric_se, all_nonparametric_se, all_nocorrection_t_stats, all_parametric_t_stats, all_nonparametric_t_stats):
    fig4, axes4 = plt.subplots(1, 3)
    fig4.tight_layout()
    fig4.set_size_inches(20,10)

    axes4[0].scatter(all_true_outcomes, all_nocorrection_estimates, color='orange', alpha=0.35)
    axes4[0].scatter(all_true_outcomes, all_parametric_estimates, color='blue', alpha=0.6)
    axes4[0].scatter(all_true_outcomes, all_nonparametric_estimates, color='skyblue', alpha=0.25)
    axes4[0].plot([0,15], [0,15], color="black")
    axes4[0].set_title("Figure 4A")
    axes4[0].set(xlabel="estimate with true outcome", ylabel="estimate with predicted outcome")

    axes4[1].scatter(all_true_se, all_nocorrection_se, color='orange', alpha=0.35)
    axes4[1].scatter(all_true_se, all_parametric_se, color='blue', alpha=0.6)
    axes4[1].scatter(all_true_se, all_nonparametric_se, color='skyblue', alpha=0.25)
    axes4[1].plot([0,0.8], [0,0.8], color="black")
    axes4[1].set_title("Figure 4B")
    axes4[1].set(xlabel="standard error with true outcome", ylabel="standard error with predicted outcome")

    axes4[2].scatter(all_true_t_stats, all_nocorrection_t_stats, color='orange', alpha=0.35)
    axes4[2].scatter(all_true_t_stats, all_parametric_t_stats, color='blue', alpha=0.6)
    axes4[2].scatter(all_true_t_stats, all_nonparametric_t_stats, color='skyblue', alpha=0.25)
    axes4[2].plot([-50,50], [-50,50], color="black")
    axes4[2].set_title("Figure 4C")
    axes4[2].set(xlabel="statistic with true outcome", ylabel="statistic with predicted outcome")

    fig4.tight_layout(pad=2.5)
    fig4.legend(['no correction', 'parametric bootstrap', 'non-parametric bootstrap', 'best_fit'], ncol=4, loc=8)

    plt.savefig("./src/plots/postpi_Fig4.png")
    plt.clf()


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


def split_data(X_test, y_test):
    test = pd.DataFrame(X_test)
    test["Spread"] = y_test

    valid = test.sample(frac=0.5)
    test = test.drop(valid.index)

    return test, valid


def postprediction_inference(X_test, y_test, prediction_model):
    all_true_outcomes, all_true_se, all_true_t_stats = [], [], []
    all_parametric_estimates, all_parametric_se, all_parametric_t_stats = [], [], []
    all_nonparametric_estimates, all_nonparametric_se, all_nonparametric_t_stats = [], [], []
    all_nocorrection_estimates, all_nocorrection_se, all_nocorrection_t_stats = [], [], []

    for i in range(250):
        if i%100 == 0:
            print("Working on Iteration: ", i)

        test_set, valid_set = split_data(X_test, y_test)
        y_test_pred = prediction_model.predict(test_set.iloc[:,:-1].values)

        if i == 0:
            figure2_plot(test_set.iloc[:,1], test_set.iloc[:,-1], y_test_pred)
            figure3_plot(y_test_pred, test_set.iloc[:,-1].values)

        relationship_model = LinearRegression(fit_intercept=False).fit(sm.add_constant(y_test_pred.reshape(-1,1)), test_set.iloc[:,-1].values)

        y_valid_pred = prediction_model.predict(valid_set.iloc[:,:-1].values)
        y_valid_corr = relationship_model.predict(sm.add_constant(y_valid_pred.reshape(-1,1)))
            
                # ------------------- true outcomes - OLS
        true_inf_model = sm.OLS(y_valid_corr, sm.add_constant(valid_set.iloc[:,1].values)).fit()
                
        all_true_outcomes.append(true_inf_model.params[1])
        all_true_se.append(true_inf_model.bse[1])
        all_true_t_stats.append(true_inf_model.tvalues[1])
                
                # ------------------- no correction method - OLS
        nocorr_inf_model = sm.OLS(y_valid_pred, sm.add_constant(valid_set.iloc[:,:-1].values)).fit()

        all_nocorrection_estimates.append(nocorr_inf_model.params[1])
        all_nocorrection_se.append(nocorr_inf_model.bse[1])
        all_nocorrection_t_stats.append(nocorr_inf_model.tvalues[1])
                
                # ------------------- parametric method
        parametric_bs_estimate, parametric_bs_se = bootstrap_(valid_set.iloc[:,1], y_valid_pred, valid_set.iloc[:,-1].values, relationship_model)
        parametric_t_stat = parametric_bs_estimate / parametric_bs_se
                
        all_parametric_estimates.append(parametric_bs_estimate)
        all_parametric_se.append(parametric_bs_se)
        all_parametric_t_stats.append(parametric_t_stat)
                
                # ------------------- non-parametric method
        nonparametric_bs_estimate, nonparametric_bs_se = bootstrap_(valid_set.iloc[:,1], y_valid_pred, valid_set.iloc[:,-1].values, relationship_model, False)
        nonparametric_t_stat = nonparametric_bs_estimate / nonparametric_bs_se
                
        all_nonparametric_estimates.append(nonparametric_bs_estimate)
        all_nonparametric_se.append(nonparametric_bs_se)
        all_nonparametric_t_stats.append(nonparametric_t_stat)

    figure4_plot(all_true_outcomes, all_true_se, all_true_t_stats, all_nocorrection_estimates, all_parametric_estimates, all_nonparametric_estimates, all_nocorrection_se, all_parametric_se, all_nonparametric_se, all_nocorrection_t_stats, all_parametric_t_stats, all_nonparametric_t_stats)
