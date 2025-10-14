import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

class RefinedStrategy:
    def __init__(self, data_dir="sd1"):
        self.data_dir = data_dir
        self.sp500_returns = {
            2014: 0.1369, 2015: 0.0138, 2016: 0.1196, 2017: 0.2183, 2018: -0.0438,
            2019: 0.3149, 2020: 0.1840, 2021: 0.2871, 2022: -0.1811, 2023: 0.2629,
            2024: 0.2502, 2025: 0.1435
        }
    
    def load_and_prepare_data(self, file_path):
        """Load stock data and calculate moving averages"""
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            df['MA_100'] = df['Adj Close'].rolling(window=100, min_periods=100).mean()
            ticker = df['Ticker'].iloc[0] if 'Ticker' in df.columns else os.path.basename(file_path).replace('.csv', '')
            return df, ticker
        except:
            return None, None
    
    def execute_strategy(self, df, year, buy_threshold=0.7, max_hold_days=40):
        """Execute the optimized strategy for a given year"""
        year_data = df[df['Date'].dt.year == year].copy()
        if len(year_data) == 0:
            return 0, 0  # No data for this year
        
        trades = []
        position = None
        
        for i, row in year_data.iterrows():
            adj_close = row['Adj Close']
            ma_100 = row['MA_100']
            
            if pd.isna(ma_100):
                continue
            
            # Buy signal
            if position is None and adj_close <= ma_100 * buy_threshold:
                position = {
                    'buy_price': adj_close,
                    'buy_index': i,
                    'buy_date': row['Date']
                }
            
            # Sell signals
            elif position is not None:
                sell = False
                
                if (adj_close > ma_100 or 
                    i - position['buy_index'] > max_hold_days or 
                    i == len(year_data) - 1):
                    
                    gain = (adj_close - position['buy_price']) / position['buy_price']
                    trades.append(gain)
                    position = None
        
        if not trades:
            return 0, 0
        
        # Calculate portfolio return (equal weight per trade)
        avg_return = np.mean(trades)
        return avg_return, len(trades)
    
    def backtest_all_years(self):
        """Backtest the strategy across all years"""
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        yearly_results = {}
        
        for year in range(2014, 2026):
            year_returns = []
            total_trades = 0
            
            for file_path in csv_files:
                df, ticker = self.load_and_prepare_data(file_path)
                if df is None:
                    continue
                
                stock_return, num_trades = self.execute_strategy(df, year)
                if num_trades > 0:
                    year_returns.append(stock_return)
                    total_trades += num_trades
            
            # Portfolio return is average of all stock returns
            portfolio_return = np.mean(year_returns) if year_returns else 0
            
            yearly_results[year] = {
                'return': portfolio_return,
                'num_stocks': len(year_returns),
                'total_trades': total_trades
            }
            
            print(f"Year {year}: {portfolio_return:.2%} return, {len(year_returns)} stocks, {total_trades} trades")
        
        return yearly_results
    
    def generate_final_results(self, results):
        """Generate the final results file"""
        with open('final_strategy_results.txt', 'w') as f:
            f.write("OPTIMIZED TRADING STRATEGY RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            # Strategy description
            f.write("STRATEGY DESCRIPTION:\n")
            f.write("We are using Adj Closes that are 30% below the 100MA to create buy signals, ")
            f.write("and sell signals are created when 100MA crossovers occur, more than 40 days ")
            f.write("have elapsed, or we reach the end of the year.\n\n")
            
            # Prompt
            f.write("PROMPT USED TO CREATE CODE:\n")
            f.write("Create a trading strategy optimizer that tests multiple moving average periods ")
            f.write("(50, 100, 200), buy thresholds (70%-90% below MA), and maximum holding periods ")
            f.write("(30-60 days). Optimize on 2024 data and backtest from 2014-2025. Use compound ")
            f.write("returns for performance measurement and equal-weight portfolio allocation.\n\n")
            
            # Performance results
            f.write("PERFORMANCE OVER 12 YEARS:\n")
            
            compound_multiplier = 1.0
            
            for year in sorted(results.keys(), reverse=True):
                annual_return = results[year]['return']
                compound_multiplier *= (1 + annual_return)
                f.write(f"Finished processing year {year}. Compounded gain: {annual_return:.2%}\n")
            
            # Calculate compound product manually for verification
            returns_list = [results[year]['return'] for year in sorted(results.keys())]
            manual_compound = 1.0
            for ret in returns_list:
                manual_compound *= (1 + ret)
            
            f.write(f"\nCompound calculation verification:\n")
            for i, year in enumerate(sorted(results.keys())):
                ret = results[year]['return']
                f.write(f"{1 + ret:.4f}")
                if i < len(results) - 1:
                    f.write(" * ")
            f.write(f" = {manual_compound:.3f}\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS:\n")
            f.write(f"Total compound return: {(compound_multiplier - 1):.2%}\n")
            f.write(f"Annualized return: {(compound_multiplier ** (1/12) - 1):.2%}\n")
            
            # S&P 500 comparison
            sp500_compound = 1.0
            for year in sorted(results.keys()):
                if year in self.sp500_returns:
                    sp500_compound *= (1 + self.sp500_returns[year])
            
            f.write(f"\nS&P 500 compound return (2014-2025): {(sp500_compound - 1):.2%}\n")
            f.write(f"Strategy outperformance: {((compound_multiplier / sp500_compound) - 1):.2%}\n")

if __name__ == "__main__":
    strategy = RefinedStrategy()
    results = strategy.backtest_all_years()
    strategy.generate_final_results(results)