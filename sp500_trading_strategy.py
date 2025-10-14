"""
S&P 500 Trading Strategy Analysis
GPT-4 Generated Code for Stock Trading Strategy Backtesting

This program implements a trading strategy that:
1. Buys when stock price is 20% below 100-day moving average
2. Sells when price rises above 100-day MA, after 40 days, or at year end
3. Analyzes performance across multiple years
4. Generates visualizations and performance reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_moving_average(prices, window=100):
    """Calculate simple moving average"""
    return prices.rolling(window=window, min_periods=window).mean()

def identify_signals(df, year=2024):
    """
    Identify buy and sell signals for a given year
    
    Buy signal: Adj Close <= 0.8 * 100-day MA
    Sell signals: 
    - Adj Close > 100-day MA
    - More than 40 trading days since buy
    - Last trading day of the year
    """
    # Filter data for the specified year
    df['Date'] = pd.to_datetime(df['Date'])
    year_data = df[df['Date'].dt.year == year].copy()
    
    if len(year_data) < 100:  # Need at least 100 days for MA calculation
        return pd.DataFrame()
    
    # Calculate 100-day moving average
    year_data = year_data.sort_values('Date').reset_index(drop=True)
    year_data['MA_100'] = calculate_moving_average(year_data['Adj Close'])
    
    # Calculate percentage below MA
    year_data['Pct_Below_MA'] = ((year_data['Adj Close'] - year_data['MA_100']) / year_data['MA_100'] * 100)
    
    # Identify buy signals (20% below MA)
    year_data['Buy_Signal'] = (year_data['Adj Close'] <= 0.8 * year_data['MA_100']) & (~year_data['MA_100'].isna())
    
    trades = []
    position = None  # Track current position
    
    for i, row in year_data.iterrows():
        if position is None and row['Buy_Signal']:
            # Open new position
            position = {
                'ticker': row['Ticker'],
                'buy_date': row['Date'],
                'buy_price': row['Adj Close'],
                'pct_below_ma': row['Pct_Below_MA'],
                'buy_index': i
            }
        
        elif position is not None:
            # Check sell conditions
            days_held = i - position['buy_index']
            is_above_ma = row['Adj Close'] > row['MA_100']
            is_40_days = days_held >= 40
            is_last_day = i == len(year_data) - 1
            
            if is_above_ma or is_40_days or is_last_day:
                # Close position
                trade = {
                    'Ticker': position['ticker'],
                    'Buy Date': position['buy_date'],
                    'Buy Price': position['buy_price'],
                    '% Below 100-day MA at Buy': position['pct_below_ma'],
                    'Sell Price': row['Adj Close'],
                    'Sell Date': row['Date'],
                    'Percentage Gain': ((row['Adj Close'] - position['buy_price']) / position['buy_price'] * 100),
                    'Trading Days Held': days_held
                }
                trades.append(trade)
                position = None
    
    return pd.DataFrame(trades)

def process_stock_file(file_path, year=2024):
    """Process a single stock CSV file"""
    try:
        df = pd.read_csv(file_path)
        if 'Ticker' not in df.columns and 'ticker' in df.columns:
            df['Ticker'] = df['ticker']
        elif 'Ticker' not in df.columns:
            # Extract ticker from filename
            ticker = os.path.basename(file_path).replace('.csv', '')
            df['Ticker'] = ticker
        
        return identify_signals(df, year)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return pd.DataFrame()

def create_stock_plot(df, trades_df, ticker, year, output_dir):
    """Create plot for a stock showing price, MA, and buy/sell signals"""
    try:
        # Filter data for the year
        df['Date'] = pd.to_datetime(df['Date'])
        year_data = df[df['Date'].dt.year == year].copy()
        
        if len(year_data) < 100:
            return
        
        year_data = year_data.sort_values('Date').reset_index(drop=True)
        year_data['MA_100'] = calculate_moving_average(year_data['Adj Close'])
        
        plt.figure(figsize=(12, 8))
        
        # Plot price and moving average
        plt.plot(year_data['Date'], year_data['Adj Close'], label='Adj Close Price', linewidth=1)
        plt.plot(year_data['Date'], year_data['MA_100'], label='100-day MA', linewidth=1, alpha=0.7)
        
        # Add buy and sell signals
        if not trades_df.empty:
            stock_trades = trades_df[trades_df['Ticker'] == ticker]
            for _, trade in stock_trades.iterrows():
                plt.axvline(x=trade['Buy Date'], color='green', linestyle='--', alpha=0.7, label='Buy' if _ == stock_trades.index[0] else "")
                plt.axvline(x=trade['Sell Date'], color='red', linestyle='--', alpha=0.7, label='Sell' if _ == stock_trades.index[0] else "")
        
        plt.title(f'{ticker} - {year} Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{ticker}_{year}_signals.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating plot for {ticker}: {e}")

def backtest_years(data_dir, years_range):
    """Backtest strategy across multiple years"""
    results = {}
    
    for year in years_range:
        print(f"Backtesting year {year}...")
        all_trades = []
        
        # Process all CSV files
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(data_dir, filename)
                trades = process_stock_file(file_path, year)
                if not trades.empty:
                    all_trades.append(trades)
        
        if all_trades:
            year_trades = pd.concat(all_trades, ignore_index=True)
            
            # Calculate performance metrics
            total_trades = len(year_trades)
            winning_trades = len(year_trades[year_trades['Percentage Gain'] > 0])
            avg_gain = year_trades['Percentage Gain'].mean()
            total_return = year_trades['Percentage Gain'].sum()
            
            # Calculate compounded return (assuming equal position sizes)
            if total_trades > 0:
                compound_return = np.prod(1 + year_trades['Percentage Gain'] / 100) - 1
            else:
                compound_return = 0
            
            results[year] = {
                'Total Trades': total_trades,
                'Winning Trades': winning_trades,
                'Win Rate (%)': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                'Average Gain (%)': avg_gain,
                'Total Return (%)': total_return,
                'Compound Return (%)': compound_return * 100
            }
        else:
            results[year] = {
                'Total Trades': 0,
                'Winning Trades': 0,
                'Win Rate (%)': 0,
                'Average Gain (%)': 0,
                'Total Return (%)': 0,
                'Compound Return (%)': 0
            }
    
    return results

def process_sd1_files(sd1_dir, output_dir):
    """Process each file in sd1 directory and create individual outcome files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(sd1_dir):
        if filename.endswith('.csv'):
            print(f"Processing {filename} for individual analysis...")
            file_path = os.path.join(sd1_dir, filename)
            ticker = filename.replace('.csv', '')
            
            try:
                df = pd.read_csv(file_path)
                df['Ticker'] = ticker
                
                # Process for multiple years
                all_outcomes = []
                for year in range(2014, 2026):
                    trades = identify_signals(df, year)
                    if not trades.empty:
                        trades['Year'] = year
                        all_outcomes.append(trades)
                
                if all_outcomes:
                    outcomes_df = pd.concat(all_outcomes, ignore_index=True)
                    output_file = os.path.join(output_dir, f'{ticker}_outcomes.csv')
                    outcomes_df.to_csv(output_file, index=False)
                    print(f"Saved outcomes for {ticker}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def play():
    """Main function implementing the trading strategy analysis"""
    
    # Print GPT version
    print("=" * 60)
    print("S&P 500 Trading Strategy Analysis")
    print("Generated by: GPT-4 (Claude-3.5-Sonnet)")
    print("=" * 60)
    
    # Set parameters
    theYear = 2024
    
    # Define directories
    base_dir = r"c:\Users\G00424185@atu.ie\OneDrive - Atlantic TU\Full Stack\Full-Stack_Week3_Lab"
    
    # Use sd1 as the stock data directory since sp500_stock_data doesn't exist
    stock_data_dir = os.path.join(base_dir, "sd1")
    plots_dir = os.path.join(base_dir, "plots")
    sd1_outcomes_dir = os.path.join(base_dir, "sd1_outcomes")
    
    if not os.path.exists(stock_data_dir):
        print(f"Error: Stock data directory not found: {stock_data_dir}")
        return
    
    print(f"Processing stock data from: {stock_data_dir}")
    print(f"Target year: {theYear}")
    
    # Process all stocks for the target year
    print(f"\nProcessing stocks for year {theYear}...")
    all_trades = []
    stocks_with_trades = []
    
    csv_files = [f for f in os.listdir(stock_data_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files to process")
    
    for i, filename in enumerate(csv_files):
        if i % 50 == 0:  # Progress indicator
            print(f"Processed {i}/{len(csv_files)} files...")
        
        file_path = os.path.join(stock_data_dir, filename)
        trades = process_stock_file(file_path, theYear)
        
        if not trades.empty:
            all_trades.append(trades)
            ticker = filename.replace('.csv', '')
            stocks_with_trades.append((ticker, file_path))
    
    print(f"Completed processing all files.")
    
    if not all_trades:
        print("No trades found for the specified year.")
        return
    
    # Combine all trades
    trades_df = pd.concat(all_trades, ignore_index=True)
    trades_df = trades_df.sort_values('Buy Date').reset_index(drop=True)
    
    # Save performance CSV
    performance_file = os.path.join(base_dir, f"{theYear}_perf.csv")
    trades_df.to_csv(performance_file, index=False)
    print(f"\nSaved performance data to: {performance_file}")
    print(f"Total trades found: {len(trades_df)}")
    
    # Create plots for stocks with trades
    print(f"\nCreating plots for {len(stocks_with_trades)} stocks with trades...")
    os.makedirs(plots_dir, exist_ok=True)
    
    for ticker, file_path in stocks_with_trades:
        try:
            df = pd.read_csv(file_path)
            df['Ticker'] = ticker
            create_stock_plot(df, trades_df, ticker, theYear, plots_dir)
        except Exception as e:
            print(f"Error creating plot for {ticker}: {e}")
    
    print(f"Plots saved to: {plots_dir}")
    
    # Backtest across multiple years (2014-2025)
    print("\nBacktesting strategy across years 2014-2025...")
    backtest_results = backtest_years(stock_data_dir, range(2014, 2026))
    
    # Print performance summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY BY YEAR")
    print("=" * 80)
    
    for year, metrics in backtest_results.items():
        print(f"\nYear {year}:")
        print(f"  Total Trades: {metrics['Total Trades']}")
        print(f"  Winning Trades: {metrics['Winning Trades']}")
        print(f"  Win Rate: {metrics['Win Rate (%)']:.1f}%")
        print(f"  Average Gain: {metrics['Average Gain (%)']:.2f}%")
        print(f"  Compound Return: {metrics['Compound Return (%)']:.2f}%")
    
    # Process SD1 files for individual analysis
    print(f"\nProcessing individual stock files from sd1 directory...")
    process_sd1_files(stock_data_dir, sd1_outcomes_dir)
    print(f"Individual stock outcomes saved to: {sd1_outcomes_dir}")
    
    # Create strategy description and results file
    strategy_description = """
TRADING STRATEGY DESCRIPTION
============================

Strategy Name: Mean Reversion Strategy (20% Below 100-day MA)

Entry Criteria:
- Stock's adjusted close price is at least 20% below its 100-day moving average
- Sufficient historical data available (minimum 100 days)

Exit Criteria (First condition met):
1. Stock price rises above the 100-day moving average
2. Position held for more than 40 trading days
3. Last trading day of the year

Position Sizing: Equal weight for all positions
Risk Management: Automatic exit after 40 days maximum hold period

ORIGINAL PROMPT:
===============
Write a Python program with a function called play() that implements a comprehensive 
S&P 500 stock trading strategy analysis system. The system should process CSV files 
containing daily stock data, identify buy/sell signals based on mean reversion 
criteria, backtest the strategy across multiple years, generate performance reports 
and visualizations, and provide detailed analysis of trading outcomes.

Key Features:
- Mean reversion strategy (buy 20% below 100-day MA)
- Multi-year backtesting (2014-2025)
- Performance visualization and reporting
- Individual stock analysis
- Comprehensive trade tracking and metrics

IMPLEMENTATION NOTES:
====================
- Uses pandas for data manipulation and analysis
- Matplotlib for visualization generation
- Implements proper error handling for missing data
- Calculates compound returns and win rates
- Generates individual stock outcome files
- Creates visual plots with buy/sell signal markers
"""
    
    results_file = os.path.join(base_dir, "strategy_results.txt")
    with open(results_file, 'w') as f:
        f.write(strategy_description)
        f.write(f"\n\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        f.write(f"\nGPT Version: GPT-4 (Claude-3.5-Sonnet)")
        f.write(f"\nTarget Year: {theYear}")
        f.write(f"\nTotal Files Processed: {len(csv_files)}")
        f.write(f"\nStocks with Trades: {len(stocks_with_trades)}")
        f.write(f"\nTotal Trades Generated: {len(trades_df)}")
    
    print(f"\nStrategy description and results saved to: {results_file}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"✓ Processed {len(csv_files)} stock files")
    print(f"✓ Generated {len(trades_df)} trades for {theYear}")
    print(f"✓ Created plots for {len(stocks_with_trades)} stocks")
    print(f"✓ Backtested {len(backtest_results)} years (2014-2025)")
    print(f"✓ Performance data saved to: {performance_file}")
    print(f"✓ Plots saved to: {plots_dir}")
    print(f"✓ Individual outcomes saved to: {sd1_outcomes_dir}")
    print(f"✓ Strategy results saved to: {results_file}")
    
    # Show sample of trades
    if len(trades_df) > 0:
        print(f"\nSample of trades for {theYear}:")
        print(trades_df.head().to_string(index=False))
        
        avg_gain = trades_df['Percentage Gain'].mean()
        win_rate = len(trades_df[trades_df['Percentage Gain'] > 0]) / len(trades_df) * 100
        print(f"\n{theYear} Performance Summary:")
        print(f"Average Gain: {avg_gain:.2f}%")
        print(f"Win Rate: {win_rate:.1f}%")

if __name__ == "__main__":
    play()