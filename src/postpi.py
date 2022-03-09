import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random
import pandas as pd

from sklearn.linear_model import LinearRegression


def figure2_plot(covariate, y_test, y_pred, covariate_name):
    """
    Plots comparison between covariate of interest and predicted/observed outcomes

    Parameters
    ----------
    covariate: pandas.Series
        column of our covariate of interest in test set
    y_test: pandas.Series
        observed Spread values from test set
    y_pred: pandas.Series
        predicted Spread values from prediction model on test set
    covariate_name: str
        name of covariate to label axis

    Returns
    -------
    None
        prints seaborn multiple scatterplot & outputs to plots/ folder
    """
    # creating the plot layout
    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Figure 2 Plots")
    fig.set_size_inches(15, 5)

    # creating a polyfit for observed values 
    m, b = np.polyfit(covariate, y_test, 1)
    # plotting the points for observed values and line of best fit
    axes[0].scatter(x=covariate, y=y_test, c="blue", alpha=0.35)
    axes[0].plot(covariate, m*covariate + b, c="red")
    axes[0].set(xlabel=f"{covariate_name} x", ylabel="Observed outcomes y")

    # creating a polyfit for predictions values
    m, b = np.polyfit(covariate, y_pred, 1)
    # plotting the points for predicted value and line of best fit
    axes[1].scatter(x=covariate, y=y_pred, c="red", alpha=0.35)
    axes[1].plot(covariate, m*covariate + b, c="red")
    axes[1].set(xlabel=f"{covariate_name} x", ylabel="Predicted outcomes y")

    # exporting to plots/ folder
    plt.savefig("src/plots/postpi_Fig2.png")
    plt.clf()


def figure3_plot(y_pred, y_pred_baseline, y_test, y_test_baseline):
    """
    Plots comparison between the predictions of our baseline model and nn model

    Parameters
    ----------
    y_pred: pandas.Series
        predictions on test set of nn model
    y_pred_baseline: pandas.Series
        predictions on test set of baseline linear regression model
    y_test: pandas.Series
        observed Spread values on test set used in nn model development
    y_test_baseline: str
        observed Spread values on test set used in baseline model development

    Returns
    -------
        None
            prints seaborn multiple scatterplot & outputs to plots/ folder
    """
    # creating the plot layout
    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Figure 3 Plots")
    fig.set_size_inches(15, 5)
    
    # creating a polyfit for MLPRegressor's predictions and observed outcomes
    m, b = np.polyfit(y_test, y_pred, 1)
    # plotting the points and line of best fit for NN predicted outcomes and observed outcomes
    axes[0].scatter(x=y_test, y=y_pred, c="purple", alpha=0.35)
    axes[0].plot(y_test, m*y_test + b, c="red")
    axes[0].set(xlabel="y-observed nn", ylabel="nn y-predicted")
    axes[0].set_title("MLPRegressor")

    # creating a polyfit for baselien model's predictions and observed outcomes
    m, b = np.polyfit(y_test_baseline, y_pred_baseline, 1)
    # plotting the points and line of best fit for baseline model's predicted outcomes and observed outcomes
    axes[1].scatter(x=y_test_baseline, y=y_pred_baseline, c="purple", alpha=0.35)
    axes[1].plot(y_test_baseline, m*y_test_baseline + b, c="red")
    axes[1].set(xlabel="y-observed baseline", ylabel="baseline y-predicted")
    axes[1].set_title("Linear Regression")

    # exporting to plots/ folder
    plt.savefig("./src/plots/postpi_Fig3.png")
    plt.clf()


def figure4_plot(all_true_outcomes, all_true_se, all_true_t_stats, all_nocorrection_estimates, all_parametric_estimates, all_nonparametric_estimates, all_nocorrection_se, all_parametric_se, all_nonparametric_se, all_nocorrection_t_stats, all_parametric_t_stats, all_nonparametric_t_stats):
    """
    Plots comparison after using postpi to correct inference on covariate of interest

    Parameters
    ----------
    all_true_outcomes, all_true_se, all_true_t_stats: List
        beta estimate, beta standard errors, beta t statistics using observed values in dataset

    all_nocorrection_estimates, all_nocorrection_se, all_nocorrection_t_stats: List
        beta estimate, beta standard errors, beta t statistics using observed values with only covariate of interest in dataset
    
    all_parametric_estimates, all_parametric_se, all_parametric_t_stats: List
        beta estimate, beta standard errors, beta t statistics using observed values with only covariate of interest in dataset
        using parametric bootstrap approach
    
    all_nonparametric_estimates, all_nonparametric_se, all_nonparametric_t_stats: List
        beta estimate, beta standard errors, beta t statistics using observed values with only covariate of interest in dataset
        using non-parametric bootstrap approach

    Returns
    -------
        None
            prints seaborn multiple scatterplot & outputs to plots/ folder
    """
    # creating the plot layout
    fig4, axes4 = plt.subplots(1, 3)
    fig4.tight_layout()
    fig4.suptitle("Figure 4 Plots")
    fig4.set_size_inches(20,10)

    # creating the beta estimate plots for all three methods for easy comparison
    axes4[0].scatter(all_true_outcomes, all_nocorrection_estimates, color='orange', alpha=0.35)
    axes4[0].scatter(all_true_outcomes, all_parametric_estimates, color='blue', alpha=0.6)
    axes4[0].scatter(all_true_outcomes, all_nonparametric_estimates, color='skyblue', alpha=0.25)
    axes4[0].plot([0,15], [0,15], color="black")
    axes4[0].set_title("Beta Estimates")
    axes4[0].set(xlabel="estimate with true outcome", ylabel="estimate with predicted outcome")

    # creating the beta standard error plots for all three methods for easy comparison
    axes4[1].scatter(all_true_se, all_nocorrection_se, color='orange', alpha=0.35)
    axes4[1].scatter(all_true_se, all_parametric_se, color='blue', alpha=0.6)
    axes4[1].scatter(all_true_se, all_nonparametric_se, color='skyblue', alpha=0.25)
    axes4[1].plot([0,0.8], [0,0.8], color="black")
    axes4[1].set_title("Standard Error")
    axes4[1].set(xlabel="standard error with true outcome", ylabel="standard error with predicted outcome")

    # creating the t-statistic plots for all three methods for easy comparison
    axes4[2].scatter(all_true_t_stats, all_nocorrection_t_stats, color='orange', alpha=0.35)
    axes4[2].scatter(all_true_t_stats, all_parametric_t_stats, color='blue', alpha=0.6)
    axes4[2].scatter(all_true_t_stats, all_nonparametric_t_stats, color='skyblue', alpha=0.25)
    axes4[2].plot([-50,50], [-50,50], color="black")
    axes4[2].set_title("T-statistic")
    axes4[2].set(xlabel="statistic with true outcome", ylabel="statistic with predicted outcome")

    fig4.tight_layout(pad=2.5)
    fig4.legend(['no correction', 'parametric bootstrap', 'non-parametric bootstrap', 'accurate_line'], ncol=4, loc=8)

    # exporting to plots/ folder
    plt.savefig("./src/plots/postpi_Fig4.png")
    plt.clf()


def bootstrap_(x_val, y_val_pred, relationship_model, param=True, B=100):
    """
    Plots comparison between the predictions of our baseline model and nn model

    Parameters
    ----------
    x_val: pandas.DataFrame
        validation feature matrix with covariate of interest
    y_val_pred: pandas.Series
        predictions on validation set
    relationship_model: sklearn.linear_model._base.LinearRegression
        model that relates predicted outcomes to observed outcomes
    param: boolean
        variable to conduct parametric bootstrap or non-parametric bootstrap
    B: int
        number of bootstrap iterations to run

    Returns
    -------
        beta_hat_boot: int
            beta estimate from bootstrapping approach
        se_hat_boot: int
            standard error from bootstrapping based on param/non-param approach
        pval_beta_estimators: int
            median of our p-values of our beta estimates from bootstrap approach
    """
    # variable to hold our sampled bootstrap covariate-outcome pairs
    bootstrap_sample_pairs = []

    # combining X_val and y_val into one dataframe to sample accurately
    for i in range(len(x_val)):
        bootstrap_sample_pairs.append([x_val.values[i]] + [y_val_pred[i]])

    # variables to hold the values we need to have 
    beta_estimators = []
    se_beta_estimators = []
    pval_beta_estimators = []
    
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
        pval_beta_estimators.append(inf_model.pvalues[1])
        
    # take median of all beta estimates
    beta_hat_boot = np.median(beta_estimators)
    # depending on param/non-param calculate SE of beta estimate accordingly
    se_hat_boot = None
    if param:
        se_hat_boot = np.median(se_beta_estimators)
    else:
        se_hat_boot = np.std(beta_estimators)
    
    return beta_hat_boot, se_hat_boot, np.median(pval_beta_estimators)


def split_data(X_test, y_test):
    """
    splits our original test data into test and validation sets

    Parameters
    ----------
    X_test: pandas.DataFrame
        original test data from year 2018-2021
    y_test: pandas.Series
        observed spread from X_test

    Returns
    -------
        test: pandas.DataFrame
            remaining observations for test set
        valid: pandas.DataFrame
            new observations in validation set
    """
    # convert X_test to dataframe
    test = pd.DataFrame(X_test)
    # add observed outcomes again
    test["Spread"] = y_test

    # sample 50% randomly for validation set
    valid = test.sample(frac=0.5)

    # rest of values not sampled goes into test set
    test = test.drop(valid.index)

    return test, valid


def postprediction_inference(X_test, y_test, prediction_model, y_test_baseline, y_pred_baseline):
    """
    conducts our post-prediction inference on our covariate of interest
        follows similar method to Q1 report and research studied in Q1

    Parameters
    ----------
    X_test: pandas.DataFrame
        original test data from year 2018-2021
    y_test: pandas.Series
        observed spread from X_test
    prediction_model: sklearn.neural_network.MLPRegressor
        NN model that uses all covariates to predict response variable in Spread
    y_test_baseline: pandas.Series
        observed outcomes from baseline model's test set
    y_pred_baseline: pandas.Series
        predicted outcomes from baseline model's test set

    Returns
    -------
        None
            prints multiple seaborn scatterplot & outputs to plots/ folder
    """
    # variables to hold all information for the true estimates/standard errors/t-statistics/p-values
    all_true_outcomes, all_true_se, all_true_t_stats, all_true_p_values = [], [], [], []
    # variables to hold all information for the parametric bootstrap estimates/standard errors/t-statistics/p-values
    all_parametric_estimates, all_parametric_se, all_parametric_t_stats, all_parametric_p_values = [], [], [], []
    # variables to hold all information for the non-parametric bootstrap estimates/standard errors/t-statistics/p-values
    all_nonparametric_estimates, all_nonparametric_se, all_nonparametric_t_stats, all_nonparametric_p_values = [], [], [], []
    # variables to hold all information for the no correction estimates/standard errors/t-statistics/p-values
    all_nocorrection_estimates, all_nocorrection_se, all_nocorrection_t_stats, all_nocorrection_p_values = [], [], [], []

    # dictionary to hold our features to try out different features in post-prediction inference
    covs = {"QBRating": 1, "TOP": -2, "RushTD": -3, 
    "SkYds": 6, "Sk": 7, "Int": 8, "1stD": 2, "Temperature": -1, "PassTD": 9}
    # variable that can be changed to analyze different covariates
    cov_of_int = "PassTD"

    # run the simulation 1000 times
    for i in range(1000):
        print(f"Working on Iteration {i}", end="\r")

        # split the test data into 50/50 new validation and test set randomly each iteration
        test_set, valid_set = split_data(X_test, y_test)
        # obtain predictions using all covariates
        y_pred_nn = prediction_model.predict(test_set.iloc[:,:-1].values)

        # only plot these figures on the first iteration
        if i == 0:
            figure2_plot(test_set.iloc[:,covs[cov_of_int]], test_set.iloc[:,-1], y_pred_nn, cov_of_int)
            figure3_plot(y_pred_nn, y_pred_baseline, test_set.iloc[:,-1].values, y_test_baseline)

        # create a relationship model that relates predicted y values and observed y values from the test set
        relationship_model = LinearRegression(fit_intercept=False).fit(sm.add_constant(y_pred_nn.reshape(-1,1)), test_set.iloc[:,-1].values)

        # use the prediction model to predict y values for the validation set
        y_valid_pred = prediction_model.predict(valid_set.iloc[:,:-1].values)
        # use the relationship model to correct predicted validation values and obtain "observed" validation values 
        y_valid_corr = relationship_model.predict(sm.add_constant(y_valid_pred.reshape(-1,1)))
            
        # ------------------- true outcomes - OLS
        # true inference model is a linear regression model using the covariate of interest to predict the corrected validation y values 
        true_inf_model = sm.OLS(y_valid_corr, sm.add_constant(valid_set.iloc[:,covs[cov_of_int]].values)).fit()

        # adding the values from the model to our variables        
        all_true_outcomes.append(true_inf_model.params[1])
        all_true_se.append(true_inf_model.bse[1])
        all_true_t_stats.append(true_inf_model.tvalues[1])
        all_true_p_values.append(true_inf_model.pvalues[1])
                
        # ------------------- no correction method - OLS
        # no correction model is a linear regression using the covariate of interest to predict the predicted validation y values 
        nocorr_inf_model = sm.OLS(y_valid_pred, sm.add_constant(valid_set.iloc[:,covs[cov_of_int]].values)).fit()

        # adding the values from the model to our variables
        all_nocorrection_estimates.append(nocorr_inf_model.params[1])
        all_nocorrection_se.append(nocorr_inf_model.bse[1])
        all_nocorrection_t_stats.append(nocorr_inf_model.tvalues[1])
        all_nocorrection_p_values.append(nocorr_inf_model.pvalues[1])
                
        # ------------------- parametric method
        # use bootstrap with parametric method to obtain values
        parametric_bs_estimate, parametric_bs_se, parametric_bs_pval = bootstrap_(valid_set.iloc[:,covs[cov_of_int]], y_valid_pred, relationship_model)
        parametric_t_stat = parametric_bs_estimate / parametric_bs_se
                
        # adding the values from the model to our variables
        all_parametric_estimates.append(parametric_bs_estimate)
        all_parametric_se.append(parametric_bs_se)
        all_parametric_t_stats.append(parametric_t_stat)
        all_parametric_p_values.append(parametric_bs_pval)
                
        # ------------------- non-parametric method
        # use bootstrap with non-parametric method to obtain values
        nonparametric_bs_estimate, nonparametric_bs_se, nonparametric_bs_pval = bootstrap_(valid_set.iloc[:,covs[cov_of_int]], y_valid_pred, relationship_model, False)
        nonparametric_t_stat = nonparametric_bs_estimate / nonparametric_bs_se
        
        # adding the values from the model to our variables
        all_nonparametric_estimates.append(nonparametric_bs_estimate)
        all_nonparametric_se.append(nonparametric_bs_se)
        all_nonparametric_t_stats.append(nonparametric_t_stat)
        all_nonparametric_p_values.append(nonparametric_bs_pval)

    # use the values calculated from the loop to make our plots
    # figure4_plot(all_true_outcomes, all_true_se, all_true_t_stats, all_nocorrection_estimates, all_parametric_estimates, all_nonparametric_estimates, all_nocorrection_se, all_parametric_se, all_nonparametric_se, all_nocorrection_t_stats, all_parametric_t_stats, all_nonparametric_t_stats)
    # pval_plot(all_true_p_values, all_nocorrection_p_values, all_nonparametric_p_values, all_parametric_p_values)

    hextri_estimates_df = pd.DataFrame({
        "y": all_nocorrection_estimates + all_parametric_estimates + all_nonparametric_estimates, 
        "x": all_true_outcomes + all_true_outcomes + all_true_outcomes,
        "class": ["no_correction"]*1000 + ["parametric"]*1000 + ["non-parametric"]*1000})

    hextri_se_df = pd.DataFrame({
        "y": all_nocorrection_se + all_parametric_se + all_nonparametric_se, 
        "x": all_true_se + all_true_se + all_true_se,
        "class": ["no_correction"]*1000 + ["parametric"]*1000 + ["non-parametric"]*1000})

    hextri_tstat_df = pd.DataFrame({
        "y": all_nocorrection_t_stats + all_parametric_t_stats + all_nonparametric_t_stats, 
        "x": all_true_t_stats + all_true_t_stats + all_true_t_stats,
        "class": ["no_correction"]*1000 + ["parametric"]*1000 + ["non-parametric"]*1000})

    hextri_pval_df = pd.DataFrame({
        "y": all_nocorrection_p_values + all_parametric_p_values + all_nonparametric_p_values, 
        "x": all_true_p_values + all_true_p_values + all_true_p_values,
        "class": ["no_correction"]*1000 + ["parametric"]*1000 + ["non-parametric"]*1000})


    hextri_estimates_df.to_csv("./src/hextri_data/estimates.csv", index=False)
    hextri_se_df.to_csv("./src/hextri_data/ses.csv", index=False)
    hextri_tstat_df.to_csv("./src/hextri_data/tstats.csv", index=False)
    hextri_pval_df.to_csv("./src/hextri_data/pvals.csv", index=False)




def pval_plot(all_true_p_values, all_nocorrection_p_values, all_nonparametric_p_values, all_parametric_p_values):
    """
    creates a plot similar to postpi graphs, but with p-values of the different methods

    Parameters
    ----------
    all_true_p_values: list
        actual p-values
    all_nocorrection_p_values: list
        no correction p-values
    all_nonparametric_p_values: list
        non-parametric bootstrap p-values
    all_parametric_p_values: list
        parametric bootstrap p-values

    Returns
    -------
        None
            prints seaborn scatterplot & outputs to plots/ folder
    """
    # creating the plot layout
    fig, ax = plt.subplots(1, 1)
    fig.suptitle("p-value plots")
    fig.set_size_inches(20,10)

    # plot p-values from each other different approaches for easy comparison
    ax.scatter(all_true_p_values, all_nocorrection_p_values, color='orange', alpha=0.35)
    ax.scatter(all_true_p_values, all_parametric_p_values, color='blue', alpha=0.6)
    ax.scatter(all_true_p_values, all_nonparametric_p_values, color='skyblue', alpha=0.25)
    ax.plot([min(all_true_p_values),max(all_true_p_values)], [min(all_true_p_values),max(all_true_p_values)], color="black")
    ax.set_title("p-values")
    ax.set(xlabel="p-value with true outcome", ylabel="p-value with predicted outcome")

    # export to plots/ folder
    plt.savefig("./src/plots/p-value_plot.png")
    plt.clf()
