import pandas as pd
import numpy as np
import os
import glob
from itertools import product

def load_stock_data(file_path):
    """Load and prepare stock data with moving averages"""
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Calculate moving averages
        df['MA_50'] = df['Adj Close'].rolling(window=50, min_periods=50).mean()
        df['MA_100'] = df['Adj Close'].rolling(window=100, min_periods=100).mean()
        df['MA_200'] = df['Adj Close'].rolling(window=200, min_periods=200).mean()
        
        ticker = df['Ticker'].iloc[0] if 'Ticker' in df.columns else os.path.basename(file_path).replace('.csv', '')
        return df, ticker
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def test_strategy_year(df, year, ma_period=100, buy_threshold=0.8, max_hold_days=40):
    """Test strategy for a specific year"""
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
                    'buy_index': i
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
                    'buy_date': position['buy_date'],
                    'sell_date': row['Date'],
                    'gain': gain,
                    'days_held': i - position['buy_index'],
                    'sell_reason': sell_reason
                })
                position = None
    
    return trades

def optimize_strategy():
    """Optimize strategy parameters for 2024"""
    print("Optimizing strategy for 2024...")
    
    # Parameter ranges to test
    ma_periods = [50, 100, 200]
    buy_thresholds = [0.7, 0.75, 0.8, 0.85, 0.9]
    max_hold_days = [30, 40, 50, 60]
    
    csv_files = glob.glob(os.path.join("sd1", "*.csv"))[:20]  # Test on first 20 stocks
    
    best_config = None
    best_performance = -float('inf')
    
    for ma_period, buy_threshold, max_days in product(ma_periods, buy_thresholds, max_hold_days):
        all_trades = []
        
        for file_path in csv_files:
            df, ticker = load_stock_data(file_path)
            if df is None:
                continue
            
            trades = test_strategy_year(df, 2024, ma_period, buy_threshold, max_days)
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
                    'num_trades': len(all_trades),
                    'performance_score': performance_score
                }
    
    return best_config

def backtest_strategy(config):
    """Backtest the optimized strategy from 2014-2025"""
    print(f"Backtesting strategy: MA={config['ma_period']}, Threshold={config['buy_threshold']:.0%}, Max Days={config['max_hold_days']}")
    
    csv_files = glob.glob(os.path.join("sd1", "*.csv"))
    yearly_results = {}
    
    for year in range(2014, 2026):
        all_trades = []
        
        for file_path in csv_files:
            df, ticker = load_stock_data(file_path)
            if df is None:
                continue
            
            trades = test_strategy_year(df, year, 
                                     config['ma_period'], 
                                     config['buy_threshold'], 
                                     config['max_hold_days'])
            all_trades.extend(trades)
        
        if all_trades:
            # Calculate portfolio return (equal weight per trade)
            portfolio_return = np.mean([t['gain'] for t in all_trades])
            yearly_results[year] = {
                'return': portfolio_return,
                'num_trades': len(all_trades),
                'win_rate': np.mean([t['gain'] > 0 for t in all_trades]),
                'avg_gain': portfolio_return
            }
        else:
            yearly_results[year] = {'return': 0, 'num_trades': 0, 'win_rate': 0, 'avg_gain': 0}
    
    return yearly_results

def generate_results_file(config, results):
    """Generate comprehensive results file"""
    with open('strategy_performance_results.txt', 'w') as f:
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
            annual_return = results[year]['return']
            compound_product *= (1 + annual_return)
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
        f.write(f"Total compound return: {(compound_product - 1):.2%}\n")
        f.write(f"Annualized return: {(compound_product ** (1/12) - 1):.2%}\n")
        
        # Strategy details
        f.write(f"\nSTRATEGY PARAMETERS:\n")
        f.write(f"Moving Average Period: {config['ma_period']} days\n")
        f.write(f"Buy Threshold: {config['buy_threshold']:.0%} of MA (buy when {(1-config['buy_threshold'])*100:.0f}% below)\n")
        f.write(f"Maximum Hold Days: {config['max_hold_days']}\n")
        f.write(f"2024 Optimization Performance Score: {config['performance_score']:.4f}\n")
        f.write(f"2024 Average Gain: {config['avg_gain']:.2%}\n")
        f.write(f"2024 Win Rate: {config['win_rate']:.1%}\n")

def main():
    """Main execution function"""
    # Step 1: Optimize for 2024
    best_config = optimize_strategy()
    
    if best_config is None:
        print("No valid configuration found!")
        return
    
    print(f"\nBest configuration found:")
    print(f"MA Period: {best_config['ma_period']}")
    print(f"Buy Threshold: {best_config['buy_threshold']:.0%} below MA")
    print(f"Max Hold Days: {best_config['max_hold_days']}")
    print(f"2024 Performance - Avg Gain: {best_config['avg_gain']:.2%}, Win Rate: {best_config['win_rate']:.1%}")
    print(f"Performance Score: {best_config['performance_score']:.4f}")
    
    # Step 2: Backtest across all years
    results = backtest_strategy(best_config)
    
    # Step 3: Generate results file
    generate_results_file(best_config, results)
    
    print(f"\nBacktest complete! Results saved to 'strategy_performance_results.txt'")
    
    # Display summary
    compound_return = 1.0
    for year in sorted(results.keys()):
        compound_return *= (1 + results[year]['return'])
    
    print(f"Total compound return (2014-2025): {(compound_return - 1):.2%}")
    print(f"Annualized return: {(compound_return ** (1/12) - 1):.2%}")

if __name__ == "__main__":
    main()