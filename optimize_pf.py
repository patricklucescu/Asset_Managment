import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optimization as op
import data_processing as dp


if __name__ == '__main__':
    # Set Parameters
    begin_date = '2003-01-01'
    end_date = '2019-03-04'
    risk_aversion = 4
    rebalancing_period = 13
    rolling_window = False  # Incorporate rolling window
    asset_removal = True  # Remove unwanted assets from the start
    HMM_parameters = [2, 0.1, 20]
    tCost = 0.002

    # Data Processing
    dtindex = pd.bdate_range(begin_date, end_date, weekmask='Fri', freq='C')
    df = pd.read_csv('asset_returns.csv', delimiter=',')
    # df = pd.read_csv('markets_new.csv', delimiter=',')
    df0 = dp.database_processing(df, dtindex, asset_removal)

    a = df0[['SPX','US 10Y']]
    df0 = a

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
                                    / (1 + (weights.loc[last, :] * all_assets_returns.loc[today, :]).sum())))  # it's just zero

    # Create TC and No TC total returns series
    turnover = turnover.fillna(0)
    total_return_noTC = (weights.shift(1) * input_returns).sum(axis=1)
    dates_index = total_return_noTC.index
    total_return_TC = np.array(total_return_noTC) - tCost*np.array(turnover.shift(1).fillna(0)).T
    total_return_TC = list(total_return_TC)[0]
    total_return_TC = pd.Series(total_return_TC, index=dates_index)

    # Remove non tradeble dates
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

    # Plotting
    plt.figure(figsize=(10,7.5))
    plt.plot(cum_returns_noTC, color="darkorchid", linewidth=2)
    plt.plot(cum_returns_TC, color='limegreen', linestyle='--', linewidth=2)
    plt.title('Geometric Sharpe : {:.3f}'.format(SR_geometric))
    plt.ylabel("Portfolio value")
    plt.show()
    plt.savefig("new_data.png", dpi=400, facecolor='aliceblue', edgecolor='k',bbox_inches='tight')
    plt.close()

    plt.figure()
    df0.pct_change().cumsum().plot()
    plt.show()
    plt.figure()
    weights.plot()
    plt.show()

    cum_spx = np.cumprod(1+input_returns['SPX'])
    cum_euro = np.cumprod(1+input_returns['Eurostoxx'])
    cum_ftse = np.cumprod(1+input_returns['FTSE'])
    cum_nik = np.cumprod(1+input_returns['Nikkei'])

    plt.figure(figsize=(10, 7.5))
    plt.plot(cum_returns_noTC, color="darkorchid", linewidth=2)
    plt.plot(cum_returns_TC, color='limegreen', linestyle='--', linewidth=2)
    plt.plot(cum_spx, color="royalblue", linewidth=2)
    plt.plot(cum_nik, color="darkorange", linewidth=2)
    plt.title('Geometric Sharpe : {:.3f}'.format(SR_geometric))
    plt.ylabel("Portfolio value")
    plt.show()

    av_weights = weights[12:]
    av_weights = av_weights.sum(0) / len(av_weights)

    plt.figure()
    plt.plot(a)
    plt.plot(pf)
    plt.show()

    plt.figure(figsize=(10, 7.5))
    plt.plot(np.cumprod(1+the_strategy),color="darkorchid", linewidth=2)
    plt.plot(np.cumprod(1+total_return_noTC),color="royalblue")
    plt.ylabel("Portfolio value")
    plt.savefig("equal.png", dpi=400, facecolor='aliceblue', edgecolor='k', bbox_inches='tight')
    plt.show()