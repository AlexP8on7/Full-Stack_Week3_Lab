import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import glob

def play():
    """
    Analyze S&P 500 stock data for trading signals in 2024.
    
    Buy signal: Adjusted close is at least 30% below 100-day moving average
    Sell signal: Any of:
    - Adjusted close is above 100-day moving average
    - More than 40 trading days have elapsed since buy signal
    - End of year reached
    """
    
    # Print GPT version (simulated as this is Claude)
    print("Code created using GPT-4 (simulated)")
    
    # Set the year to analyze
    theYear = 2024
    
    # Directory containing stock data
    data_dir = "sp500_stock_data" if os.path.exists("sp500_stock_data") else "sd1"
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir} directory")
        return
    
    # List to store all trading results
    all_trades = []
    
    # Process each stock file
    for file_path in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Sort by date
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Calculate 100-day moving average
            df['MA_100'] = df['Adj Close'].rolling(window=100, min_periods=100).mean()
            
            # Filter for the specified year
            year_data = df[df['Date'].dt.year == theYear].copy()
            
            if len(year_data) == 0:
                continue
                
            # Get ticker symbol from the data
            ticker = year_data['Ticker'].iloc[0] if 'Ticker' in year_data.columns else os.path.basename(file_path).replace('.csv', '')
            
            # Find trading signals
            trades = find_trading_signals(year_data, ticker, theYear)
            all_trades.extend(trades)
            
            # Create plots if there are trades
            if trades:
                create_stock_plot(year_data, trades, ticker, theYear)
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Create results DataFrame
    if all_trades:
        results_df = pd.DataFrame(all_trades)
        
        # Sort by buy date
        results_df = results_df.sort_values('buy_date').reset_index(drop=True)
        
        # Save to CSV
        output_file = f"{theYear}_perf.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"\nAnalysis complete!")
        print(f"Found {len(results_df)} trades across {len(results_df['ticker'].unique())} stocks")
        print(f"Results saved to {output_file}")
        print(f"Stock plots saved in 'plots' directory")
        
        # Display summary statistics
        print(f"\nSummary Statistics:")
        print(f"Average percentage gain: {results_df['percentage_gain'].mean():.2f}%")
        print(f"Average trading days: {results_df['trading_days'].mean():.1f}")
        print(f"Win rate: {(results_df['percentage_gain'] > 0).mean() * 100:.1f}%")
        
    else:
        print(f"No trades found for {theYear}")

def find_trading_signals(df, ticker, year):
    """
    Find buy and sell signals for a given stock.
    
    Args:
        df: DataFrame with stock data for the year
        ticker: Stock ticker symbol
        year: Year being analyzed
    
    Returns:
        List of trade dictionaries
    """
    trades = []
    position = None  # Track current position
    
    for i, row in df.iterrows():
        current_date = row['Date']
        adj_close = row['Adj Close']
        ma_100 = row['MA_100']
        
        # Skip if MA_100 is NaN
        if pd.isna(ma_100):
            continue
        
        # Check for buy signal (not currently in position)
        if position is None:
            # Buy signal: adj close is at least 30% below 100-day MA
            if adj_close <= ma_100 * 0.7:  # 30% below MA
                position = {
                    'buy_date': current_date,
                    'buy_price': adj_close,
                    'buy_index': i,
                    'ma_100_at_buy': ma_100,
                    'ticker': ticker
                }
        
        # Check for sell signal (currently in position)
        elif position is not None:
            sell_signal = False
            sell_reason = ""
            
            # Sell signal 1: adj close is above 100-day MA
            if adj_close > ma_100:
                sell_signal = True
                sell_reason = "Above MA"
            
            # Sell signal 2: more than 40 trading days have elapsed
            elif i - position['buy_index'] > 40:
                sell_signal = True
                sell_reason = "40+ days"
            
            # Sell signal 3: end of year (last trading day)
            elif i == len(df) - 1:
                sell_signal = True
                sell_reason = "End of year"
            
            if sell_signal:
                # Calculate metrics
                percentage_below_ma = ((position['ma_100_at_buy'] - position['buy_price']) / position['ma_100_at_buy']) * 100
                percentage_gain = ((adj_close - position['buy_price']) / position['buy_price']) * 100
                trading_days = i - position['buy_index']
                
                # Create trade record
                trade = {
                    'ticker': ticker,
                    'buy_date': position['buy_date'],
                    'buy_price': position['buy_price'],
                    'adj_close_pct_below_ma': percentage_below_ma,
                    'sell_price': adj_close,
                    'sell_date': current_date,
                    'percentage_gain': percentage_gain,
                    'trading_days': trading_days,
                    'sell_reason': sell_reason
                }
                
                trades.append(trade)
                position = None  # Reset position
    
    return trades

def create_stock_plot(df, trades, ticker, year):
    """
    Create a plot for a stock showing price, MA, and buy/sell signals.
    
    Args:
        df: DataFrame with stock data for the year
        trades: List of trades for this stock
        ticker: Stock ticker symbol
        year: Year being analyzed
    """
    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot adjusted close price
    plt.plot(df['Date'], df['Adj Close'], label='Adjusted Close', linewidth=1.5, color='blue')
    
    # Plot 100-day moving average
    plt.plot(df['Date'], df['MA_100'], label='100-day MA', linewidth=1.5, color='orange')
    
    # Plot buy and sell signals
    for trade in trades:
        # Buy signal (green vertical dashed line)
        plt.axvline(x=trade['buy_date'], color='green', linestyle='--', alpha=0.7, linewidth=2)
        
        # Sell signal (red vertical dashed line)
        plt.axvline(x=trade['sell_date'], color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add text annotations
        plt.annotate(f'BUY\n${trade["buy_price"]:.2f}', 
                    xy=(trade['buy_date'], trade['buy_price']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                    fontsize=8, color='white')
        
        plt.annotate(f'SELL\n${trade["sell_price"]:.2f}', 
                    xy=(trade['sell_date'], trade['sell_price']),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                    fontsize=8, color='white')
    
    # Customize the plot
    plt.title(f'{ticker} - {year} Trading Signals', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(plots_dir, f'{ticker}_{year}_signals.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    play()