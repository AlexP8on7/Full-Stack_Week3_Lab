The goal is to create a python project that uses downloaded daily S&P 500 ticker data stored in the sd1 project folder over the past 25 years to predict when to buy and sell stocks.
 
The code should:
 
Backtest the strategy for each year.
 
The trading strategy should be called from main. The function should be called play(year). in play it handles all the calculations for the gain etc.
Create a spreadsheet for trades with the following columns: Ticker, Buy Date, Buy Price, Sell Date, Sell Price, Percentage Profit.
Create a spreadsheet outlining all the trades that were executed in order.
 
Create charts for one trade per year and mark the buy and sell points with dotted vertical lines.
 
For each stock CSV, create another CSV with the trade outcome.
Trading strategy:

Only one stock can be bought and held at a time. before buying a new stock, the old one has to be sold.
Calculate the 20-day moving average (MA20) of the closing price to capture short-term trend direction.
 
Calculate the 20-day average daily volume (Vol20).
 
Buy Signal:
When the stock’s close price breaks above the MA20, and daily volume for that day is at least 1.5x Vol20, indicating a volume breakout confirming momentum.
 
Sell Signal:
When the close price falls back below the MA20, indicating the momentum is fading, or
more than 40 days has elapsed since the buy signal or reached the end of year.
 
Only create a sell signal after creating a correponding buy signal.
 
If a position is open at the end of the year, close it at the last available price.
 
Limit total trades per year to 50
 
Compounding individual stock returns: The original code multiplied each trade's percentage return, which compounds gains unrealistically when you have many profitable trades.
 
No position sizing: Each trade was treated as if you invested your entire portfolio, rather than dividing capital among trades.
 
Mathematical error: Multiplying percentage returns instead of using proper portfolio math.
 
Ensure that in the trading strategy that:
 
Portfolio-based approach: Start with $10,000 and allocate equal amounts per trade
 
Realistic position sizing: Divide portfolio by max trades (50) for position size
 
Proper return calculation: Calculate actual dollar gains and portfolio growth
 
Annual return: Show realistic annual percentage returns instead of compounded individual trade returns
 
 
Here’s a comprehensive table of the S&P 500 annual returns from 2000 to 2025, showing both price returns (excluding dividends) and total returns (including dividends):
 
Year Price Return (%) Total Return (%)
2000    -9.10%    -9.03%
2001    -13.04%    -11.85%
2002    -23.37%    -22.10%
2003    26.38%    28.68%
2004    8.99%    10.88%
2005    3.00%    4.91%
2006    13.62%    15.79%
2007    3.53%    5.49%
2008    -38.49%    -37.00%
2009    23.45%    26.46%
2010    12.78%    15.06%
2011    0.00%    2.11%
2012    13.41%    16.00%
2013    29.60%    32.39%
2014    11.39%    13.69%
2015    -0.73%    1.38%
2016    9.54%    11.96%
2017    19.42%    21.83%
2018    -6.24%    -4.38%
2019    28.88%    31.49%
2020    16.26%    18.40%
2021    26.89%    28.71%
2022    -18.11%    -18.11%
2023    24.25%    26.29%
2024    22.02%    25.02%
 
The output should look like this:
 
Finished processing year 2015. Compounded gain: 9.60%
etc..
Finished processing year 2025. Compounded gain: 9.60%
1.0960*1.2309*1.3611*2.0144*1.2206*1.1992*1.9372*1.1100*1.3439*1.2380*1.0337*1.3116 = 26.261
 
The Project structure should look like this:
    AIPromtingStocks
        sd1
        results
        charts
        main.py
        trading_strategy.py
 
The structure of the csv files is as follows: use this to design the trading strategy. I will upload the real csv files once the project is generated
Date,Close,Adj Close,High,Low,Open,Volume,Ticker
2000-08-11,0.7166049480438232,0.7166049480438232,0.7213006211451003,0.6846718931332565,0.7039258231211382,238056000,AAPL
2000-08-14,0.707212507724762,0.707212507724762,0.7166046525883462,0.6959420843634119,0.7151951537215286,156660000,AAPL
Note: The data for 2025 is as of September 29, 2025, and reflects year-to-date (YTD) performance.
Please generate a complete Python project based on this prompt that implements the above requirements and strategy.
