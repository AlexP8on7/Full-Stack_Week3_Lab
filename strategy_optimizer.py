import pandas as pd
import numpy as np
import os
import glob
from itertools import product
from datetime import datetime

class StrategyOptimizer:
    def __init__(self, data_dir="sd1"):
        self.data_dir = data_dir
        self.sp500_returns = {
            2014: 0.1369, 2015: 0.0138, 2016: 0.1196, 2017: 0.2183, 2018: -0.0438,
            2019: 0.3149, 2020: 0.1840, 2021: 0.2871, 2022: -0.1811, 2023: 0.2629,
            2024: 0.2502, 2025: 0.1435
        }
    
    def load_stock_data(self, ticker_file):
        """Load and prepare stock data"""
        try:
            df = pd.read_csv(ticker_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Calculate moving averages
            df['MA_50'] = df['Adj Close'].rolling(window=50, min_periods=50).mean()
            df['MA_100'] = df['Adj Close'].rolling(window=100, min_periods=100).mean()
            df['MA_200'] = df['Adj Close'].rolling(window=200, min_periods=200).mean()
            
            ticker = df['Ticker'].iloc[0] if 'Ticker' in df.columns else os.path.basename(ticker_file).replace('.csv', '')
            return df, ticker
        except:
            return None, None
    
    def test_strategy(self, df, ticker, year, ma_period=100, buy_threshold=0.8, max_hold_days=40):
        """Test a specific strategy configuration"""
        year_data = df[df['Date'].dt.year == year].copy()
        if len(year_data) == 0:
            return []
        
        ma_col = f'MA_{ma_period}'
        if ma_col not in year_data.columns:
            return []
        
        trades = []
        position = None
        
        for i, row in year_data.iterrows():
            adj_close = row['Adj Close']
            ma_value = row[ma_col]
            
            if pd.isna(ma_value):
                continue
            
            if position is None:
                # Buy signal: price below MA threshold
                if adj_close <= ma_value * buy_threshold:
                    position = {
                        'buy_date': row['Date'],
                        'buy_price': adj_close,
                        'buy_index': i,
                        'ticker': ticker
                    }
            
            elif position is not None:
                sell_signal = False
                sell_reason = ""
                
                # Sell conditions
                if adj_close > ma_value:
                    sell_signal = True
                    sell_reason = "Above MA"
                elif i - position['buy_index'] > max_hold_days:
                    sell_signal = True
                    sell_reason = f"{max_hold_days}+ days"
                elif i == len(year_data) - 1:
                    sell_signal = True
                    sell_reason = "End of year"
                
                if sell_signal:
                    gain = (adj_close - position['buy_price']) / position['buy_price']
                    trades.append({
                        'ticker': ticker,
                        'buy_date': position['buy_date'],
                        'sell_date': row['Date'],
                        'gain': gain,
                        'days_held': i - position['buy_index'],
                        'sell_reason': sell_reason
                    })
                    position = None
        
        return trades
    
    def optimize_for_year(self, year=2024):
        """Optimize strategy parameters for a specific year"""
        print(f"Optimizing strategy for {year}...")
        
        # Parameter ranges to test
        ma_periods = [50, 100, 200]
        buy_thresholds = [0.7, 0.75, 0.8, 0.85, 0.9]
        max_hold_days = [30, 40, 50, 60]
        
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))[:50]  # Test on first 50 stocks
        
        best_config = None
        best_performance = -float('inf')
        
        for ma_period, buy_threshold, max_days in product(ma_periods, buy_thresholds, max_hold_days):
            all_trades = []
            
            for file_path in csv_files:
                df, ticker = self.load_stock_data(file_path)
                if df is None:
                    continue
                
                trades = self.test_strategy(df, ticker, year, ma_period, buy_threshold, max_days)
                all_trades.extend(trades)
            
            if all_trades:
                avg_gain = np.mean([t['gain'] for t in all_trades])
                win_rate = np.mean([t['gain'] > 0 for t in all_trades])
                performance_score = avg_gain * win_rate  # Combined metric
                
                if performance_score > best_performance:
                    best_performance = performance_score
                    best_config = {
                        'ma_period': ma_period,
                        'buy_threshold': buy_threshold,
                        'max_hold_days': max_days,
                        'avg_gain': avg_gain,
                        'win_rate': win_rate,
                        'num_trades': len(all_trades)
                    }
        
        return best_config
    
    def backtest_strategy(self, config, years=range(2014, 2026)):
        """Backtest the optimized strategy across multiple years"""
        print(f"Backtesting strategy: MA={config['ma_period']}, Threshold={config['buy_threshold']:.0%}, Max Days={config['max_hold_days']}")
        
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        yearly_results = {}
        
        for year in years:
            all_trades = []
            
            for file_path in csv_files:
                df, ticker = self.load_stock_data(file_path)
                if df is None:
                    continue
                
                trades = self.test_strategy(df, ticker, year, 
                                          config['ma_period'], 
                                          config['buy_threshold'], 
                                          config['max_hold_days'])
                all_trades.extend(trades)
            
            if all_trades:
                total_gain = np.prod([1 + t['gain'] for t in all_trades]) - 1
                yearly_results[year] = {
                    'total_gain': total_gain,
                    'num_trades': len(all_trades),
                    'win_rate': np.mean([t['gain'] > 0 for t in all_trades]),
                    'avg_gain': np.mean([t['gain'] for t in all_trades])
                }
            else:
                yearly_results[year] = {'total_gain': 0, 'num_trades': 0, 'win_rate': 0, 'avg_gain': 0}
        
        return yearly_results
    
    def run_optimization_and_backtest(self):
        """Main function to optimize and backtest"""
        # Step 1: Optimize for 2024
        best_config = self.optimize_for_year(2024)
        
        if best_config is None:
            print("No valid configuration found!")
            return
        
        print(f"\nBest configuration found:")
        print(f"MA Period: {best_config['ma_period']}")
        print(f"Buy Threshold: {best_config['buy_threshold']:.0%} below MA")
        print(f"Max Hold Days: {best_config['max_hold_days']}")
        print(f"2024 Performance - Avg Gain: {best_config['avg_gain']:.2%}, Win Rate: {best_config['win_rate']:.1%}")
        
        # Step 2: Backtest across all years
        results = self.backtest_strategy(best_config)
        
        # Step 3: Generate results file
        self.generate_results_file(best_config, results)
        
        return best_config, results
    
    def generate_results_file(self, config, results):
        """Generate comprehensive results file"""
        with open('strategy_results.txt', 'w') as f:
            # Strategy description
            f.write("STRATEGY DESCRIPTION:\n")
            f.write(f"We are using Adj Closes that are {(1-config['buy_threshold'])*100:.0f}% below the {config['ma_period']}MA to create buy signals, ")
            f.write(f"and sell signals are created when {config['ma_period']}MA crossovers occur, more than {config['max_hold_days']} days have elapsed, ")
            f.write("or we reach the end of the year.\n\n")
            
            # Prompt used
            f.write("PROMPT USED:\n")
            f.write("Create a trading strategy optimizer that tests multiple moving average periods (50, 100, 200), ")
            f.write("buy thresholds (70%-90% below MA), and maximum holding periods (30-60 days). ")
            f.write("Optimize on 2024 data and backtest from 2014-2025. Use compound returns for performance measurement.\n\n")
            
            # Performance results
            f.write("PERFORMANCE OVER 12 YEARS:\n")
            compound_product = 1.0
            
            for year in sorted(results.keys(), reverse=True):
                gain_pct = results[year]['total_gain'] * 100
                compound_product *= (1 + results[year]['total_gain'])
                f.write(f"Finished processing year {year}. Compounded gain: {gain_pct:.2f}%\n")
            
            f.write(f"\nTotal compound return: {(compound_product - 1) * 100:.2f}%\n")
            f.write(f"Compound multiplier: {compound_product:.3f}\n")
            
            # Comparison with S&P 500
            f.write(f"\nS&P 500 COMPARISON:\n")
            sp500_compound = 1.0
            for year in sorted(results.keys()):
                if year in self.sp500_returns:
                    sp500_compound *= (1 + self.sp500_returns[year])
            
            f.write(f"S&P 500 compound return (2014-2025): {(sp500_compound - 1) * 100:.2f}%\n")
            f.write(f"Strategy vs S&P 500: {((compound_product / sp500_compound) - 1) * 100:.2f}% outperformance\n")

if __name__ == "__main__":
    optimizer = StrategyOptimizer()
    optimizer.run_optimization_and_backtest()