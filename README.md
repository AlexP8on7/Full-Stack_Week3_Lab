The goal is to create a python project that uses downloaded daily S&P 500 ticker data stored in the sd1 project folder over the past 25 years to predict when to buy and sell stocks.

The code should:

Backtest the strategy for each year.

The trading strategy should be called from a main.py function and take the year as the input.

Other parameters stored in variables should include indicator attributes (e.g. RSI threshold).

Create a spreadsheet for trades with the following columns: Ticker, Buy Date, Buy Price, Sell Date, Sell Price, Percentage Profit.

Adjust your variables to get a finite number of trades (e.g. 50).

Plot charts of the stocks traded, clearly marking buy and sell signals.

For each stock CSV, create another CSV with the trade outcome.

Trading strategy:

Use the Relative Strength Index (RSI) as the primary indicator with the following rules:

Calculate RSI with a period of 14 days.

Buy Signal: When RSI crosses below 30 (oversold level).

Sell Signal: When RSI crosses above 70 (overbought level).

If a position is open at the end of the year, close it at the last available price.

Limit total trades per year to a finite number (e.g. 50) to avoid overtrading.

Here’s a comprehensive table of the S&P 500 annual returns from 2000 to 2025, showing both price returns (excluding dividends) and total returns (including dividends):

[Insert the table you provided here]

Note: The data for 2025 is as of September 29, 2025, and reflects year-to-date (YTD) performance.

for the project setup, use a venv, and the requirement.txt file

Please generate a complete Python project based on this prompt that implements the above requirements and strategy.
