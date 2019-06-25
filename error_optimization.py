import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from HMM import expectation_maximization
import data_processing as dp
from matplotlib.ticker import PercentFormatter


if __name__ == '__main__':

    # SET PARAMETERS
    begin_date = '1993-01-01'
    end_date = '2002-12-31'
    rebalancing_period = 52
    rolling_window = False  # Incorporate rolling window
    asset_removal = True  # Remove unwanted assets from the start
    HMM_parameters = [[1, 0.1, 20], [2, 0.1, 20], [3, 0.1, 20], [4, 0.1, 20], [5, 0.1, 20],
                      [6, 0.1, 20], [7, 0.1, 20], [8, 0.1, 20], [9, 0.1, 20], [10, 0.1, 20]]
    # Data Processing
    dtindex = pd.bdate_range(begin_date, end_date, weekmask='Fri', freq='C')
    df = pd.read_csv('markets_new.csv', delimiter=',')
    df0 = dp.database_processing(df, dtindex, asset_removal)

    # Returns Processing
    input_returns = df0.pct_change().fillna(0)
    input_returns = input_returns.iloc[1:, :]
    rebalancing_dates = dtindex[rebalancing_period - 1::rebalancing_period]

    # Set Vectors for Errors
    mean_error = []
    max_error = []
    sign = []

    for parameters in HMM_parameters:
        maximum_error, correct_sign, error_value, counter = 0, 0, 0, 0
        K, p, iterations = parameters[0], parameters[1], parameters[2]

        for date in dtindex[rebalancing_period - 1:]:
            today = date

            if today in rebalancing_dates:
                print(today)
                print(parameters)

                # Select available assets
                returns, available_assets = dp.returns_cleaning(input_returns, df0, rolling_window, today)

                # Returns Predictions
                posteriori_prob, mu_s, cov_s, pred_ret = expectation_maximization(returns, K, iterations, p)

                # Error Processing
                actual_ret = input_returns.loc[today+1, available_assets].values
                abs_error = abs(pred_ret-actual_ret)
                counter += len(abs_error)
                error_value += sum(abs_error)
                for i in range(0, len(abs_error)):
                    if actual_ret[i]*pred_ret[i] >= 0:
                        correct_sign += 1
                maximum_error = np.max([maximum_error, np.max(abs_error)])
        max_error.append(maximum_error)
        mean_error.append(error_value/counter)
        sign.append(correct_sign/counter)

    # Plotting
    iteration = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.figure(figsize=(10, 7.5))
    plt.scatter(k, mean_error, color="darkorchid", marker='o', s=60)
    plt.xlabel("K")
    plt.ylabel("Mean absolute error")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig("kmean_abs_er.png", dpi=400, facecolor='aliceblue', edgecolor='k', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 7.5))
    plt.scatter(k, sign, color="darkorchid", marker='o', s=60)
    plt.ylabel("Sign accuracy")
    plt.xlabel("K")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig("ksign_accuracy.png", dpi=400, facecolor='aliceblue', edgecolor='k', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 7.5))
    plt.scatter(k, max_error, color="darkorchid", marker='o', s=60)
    plt.xlabel("K")
    plt.ylabel("Maximum absolute error")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig("kmax_abs_er.png", dpi=400, facecolor='aliceblue', edgecolor='k', bbox_inches='tight')
    plt.close()
