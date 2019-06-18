import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optimization as op
import data_processing as dp


if __name__ == '__main__':
    # Set Parameters
    begin_date = '1993-01-01'
    end_date = '2002-12-31'
    risk_aversion = 1
    rebalancing_period = 104
    rolling_window = False  # Incorporate rolling window
    asset_removal = True  # Remove unwanted assets from the start
    HMM_parameters = [2, 0.1, 20]
    tCost = 0.004

    # Data Processing
    dtindex = pd.bdate_range(begin_date, end_date, weekmask='Fri', freq='C')
    df = pd.read_csv('markets_new.csv', delimiter=',')
    df0 = dp.database_processing(df, dtindex, asset_removal)

    # Returns Processing
    input_returns = df0.pct_change().fillna(0)
    input_returns = input_returns.iloc[1:, :]
    rebalancing_dates = dtindex[rebalancing_period - 1::rebalancing_period]

    weights = pd.DataFrame(data=np.nan, columns=input_returns.columns, index=input_returns.index)
    turnover = pd.DataFrame(data=np.nan, columns=["1"], index=input_returns.index)
    num_assets = len(input_returns.columns)
    turnover_counter = 0

    for date in dtindex[rebalancing_period - 1:]:
        today = date
        all_assets_returns = input_returns.loc[:today, :]
        last = all_assets_returns.index[-2]

        if today in rebalancing_dates:  # re-optimize and get new weights
            print(today)
            # Select available assets
            returns, available_assets = dp.returns_cleaning(input_returns, df0, rolling_window, today)

            weights.loc[today, :] = [0] * num_assets
            weights.loc[today, available_assets] = op.optimize(returns, risk_aversion, HMM_parameters)
            if today == rebalancing_dates[0]:
                turnover.loc[today] = 1
            else:
                turnover.loc[today] = sum(abs(weights.loc[today]-weights.loc[last, :] * (1 + all_assets_returns.loc[today, :]) \
                                    / (1 + (weights.loc[last, :] * all_assets_returns.loc[today, :]).sum())))
            turnover_counter += turnover.loc[today]

        else:  # no re-optimization, re-balance the weights

            weights.loc[today, :] = weights.loc[last, :] * (1 + all_assets_returns.loc[today, :]) \
                                    / (1 + (weights.loc[last, :] * all_assets_returns.loc[today, :]).sum())
            turnover.loc[today] = sum(abs(weights.loc[today]-weights.loc[last, :] * (1 + all_assets_returns.loc[today, :]) \
                                    / (1 + (weights.loc[last, :] * all_assets_returns.loc[today, :]).sum()))) # it's just zero

    # Create TC and No TC total returns series
    turnover = turnover.fillna(0)
    total_return_noTC = (weights.shift(1) * input_returns).sum(axis=1)
    dates_index = total_return_noTC.index
    total_return_TC = np.array(total_return_noTC) - tCost*np.array(turnover.shift(1).fillna(0)).T
    total_return_TC = list(total_return_TC)[0]
    total_return_TC = pd.Series(total_return_TC, index=dates_index)

    # Remove non tradable dates
    total_return_noTC = total_return_noTC[dtindex[rebalancing_period:]]
    total_return_TC = total_return_TC[dtindex[rebalancing_period:]]


    cum_returns_noTC = np.cumprod(1+total_return_noTC)
    cum_returns_TC = np.cumprod(1 + total_return_TC)

    # Get Statistics
    annualizationFactor = 52
    nPeriods = len(total_return_noTC)  # number of weeks in our dataset
    FinalPfValTotalRet = np.prod(1 + total_return_noTC)
    GeomAvgTotalReturn_weekly = (FinalPfValTotalRet ** (1 / nPeriods) - 1)
    GeomAvgTotalReturn_annually = 100 * ((1 + GeomAvgTotalReturn_weekly) ** annualizationFactor - 1)
    StdReturns_weekly = np.std(total_return_noTC)
    SR_geometric = np.sqrt(annualizationFactor)*GeomAvgTotalReturn_weekly / StdReturns_weekly
    av_turnover = 100 * turnover_counter / len(rebalancing_dates)
    StdReturns_annually = 100*StdReturns_weekly * np.sqrt(annualizationFactor)


    plt.figure(figsize=(10,7.5))
    plt.plot(cum_returns_noTC, color="darkorchid", linewidth=2)
    plt.plot(cum_returns_TC, color='limegreen', linestyle='--', linewidth=2)
    plt.title('Geometric Sharpe : {:.3f}'.format(SR_geometric))
    plt.ylabel("Cumulative Returns")
    plt.savefig("annually.png", dpi=400, facecolor='aliceblue', edgecolor='k',bbox_inches='tight')
    plt.close()






    plt.figure
    df0.pct_change().cumsum().plot()

    plt.figure
    weights.plot()

    plt.show()
