import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

def load_and_prepare_data(file_path):
    """Load ABBV data and calculate moving averages"""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate 100-day moving average (as per strategy results)
    df['MA_100'] = df['Adj Close'].rolling(window=100, min_periods=100).mean()
    
    return df

def find_buy_sell_signals(df, year=2015, buy_threshold=0.7, max_hold_days=40):
    """Find buy and sell signals for the specified year"""
    # Filter data for the specified year
    year_data = df[df['Date'].dt.year == year].copy().reset_index(drop=True)
    
    if len(year_data) == 0:
        return year_data, [], []
    
    buy_signals = []
    sell_signals = []
    position = None
    
    for i, row in year_data.iterrows():
        adj_close = row['Adj Close']
        ma_value = row['MA_100']
        
        if pd.isna(ma_value):
            continue
        
        if position is None:
            # Buy signal: price 30% below 100MA (threshold = 0.7)
            if adj_close <= ma_value * buy_threshold:
                buy_signals.append({
                    'date': row['Date'],
                    'price': adj_close,
                    'index': i
                })
                position = {
                    'buy_date': row['Date'],
                    'buy_price': adj_close,
                    'buy_index': i
                }
        
        elif position is not None:
            sell_signal = False
            sell_reason = ""
            
            # Sell conditions
            if adj_close > ma_value:  # Price crosses above MA
                sell_signal = True
                sell_reason = "Above MA"
            elif i - position['buy_index'] >= max_hold_days:  # Max holding period
                sell_signal = True
                sell_reason = f"{max_hold_days}+ days"
            elif i == len(year_data) - 1:  # End of year
                sell_signal = True
                sell_reason = "End of year"
            
            if sell_signal:
                sell_signals.append({
                    'date': row['Date'],
                    'price': adj_close,
                    'index': i,
                    'reason': sell_reason,
                    'buy_date': position['buy_date'],
                    'buy_price': position['buy_price'],
                    'gain': (adj_close - position['buy_price']) / position['buy_price']
                })
                position = None
    
    return year_data, buy_signals, sell_signals

def create_visualization(year_data, buy_signals, sell_signals, ticker="ABBV"):
    """Create the buy/sell signals visualization"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot price and moving average
    ax.plot(year_data['Date'], year_data['Adj Close'], 
            label=f'{ticker} Adj Close', linewidth=2, color='blue', alpha=0.8)
    ax.plot(year_data['Date'], year_data['MA_100'], 
            label='100-Day MA', linewidth=2, color='orange', alpha=0.8)
    
    # Plot buy threshold line (30% below MA)
    buy_threshold_line = year_data['MA_100'] * 0.7
    ax.plot(year_data['Date'], buy_threshold_line, 
            label='Buy Threshold (30% below MA)', linewidth=1, 
            color='green', linestyle='--', alpha=0.6)
    
    # Plot buy signals
    if buy_signals:
        buy_dates = [signal['date'] for signal in buy_signals]
        buy_prices = [signal['price'] for signal in buy_signals]
        ax.scatter(buy_dates, buy_prices, 
                  color='green', marker='^', s=100, 
                  label=f'Buy Signals ({len(buy_signals)})', zorder=5)
    
    # Plot sell signals with different colors based on reason
    if sell_signals:
        sell_dates = [signal['date'] for signal in sell_signals]
        sell_prices = [signal['price'] for signal in sell_signals]
        
        # Separate by sell reason for different colors
        above_ma_dates = [s['date'] for s in sell_signals if 'Above MA' in s['reason']]
        above_ma_prices = [s['price'] for s in sell_signals if 'Above MA' in s['reason']]
        
        max_days_dates = [s['date'] for s in sell_signals if 'days' in s['reason']]
        max_days_prices = [s['price'] for s in sell_signals if 'days' in s['reason']]
        
        end_year_dates = [s['date'] for s in sell_signals if 'End of year' in s['reason']]
        end_year_prices = [s['price'] for s in sell_signals if 'End of year' in s['reason']]
        
        if above_ma_dates:
            ax.scatter(above_ma_dates, above_ma_prices, 
                      color='red', marker='v', s=100, 
                      label=f'Sell: Above MA ({len(above_ma_dates)})', zorder=5)
        
        if max_days_dates:
            ax.scatter(max_days_dates, max_days_prices, 
                      color='purple', marker='v', s=100, 
                      label=f'Sell: Max Days ({len(max_days_dates)})', zorder=5)
        
        if end_year_dates:
            ax.scatter(end_year_dates, end_year_prices, 
                      color='black', marker='v', s=100, 
                      label=f'Sell: End Year ({len(end_year_dates)})', zorder=5)
    
    # Formatting
    ax.set_title(f'{ticker} Trading Strategy - Buy/Sell Signals for 2015\n'
                'Strategy: Buy when 30% below 100MA, Sell when above MA or after 40 days', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # Add statistics text box
    if buy_signals and sell_signals:
        total_trades = len(sell_signals)
        profitable_trades = sum(1 for s in sell_signals if s['gain'] > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        avg_gain = np.mean([s['gain'] for s in sell_signals]) if sell_signals else 0
        
        stats_text = f"""2015 Strategy Performance:
Total Trades: {total_trades}
Win Rate: {win_rate:.1%}
Avg Gain: {avg_gain:.2%}
Profitable: {profitable_trades}/{total_trades}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join('plots', f'{ticker}_2015_signals.png')
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")
    
    plt.show()
    
    return fig

def print_trade_details(buy_signals, sell_signals, ticker="ABBV"):
    """Print detailed trade information"""
    print("\n" + "="*80)
    print(f"DETAILED TRADE ANALYSIS FOR {ticker} 2015")
    print("="*80)
    
    if not buy_signals or not sell_signals:
        print("No trades found for 2015")
        return
    
    print(f"\nTotal Buy Signals: {len(buy_signals)}")
    print(f"Total Sell Signals: {len(sell_signals)}")
    
    print("\nTRADE DETAILS:")
    print("-" * 80)
    
    for i, sell in enumerate(sell_signals, 1):
        days_held = (sell['date'] - sell['buy_date']).days
        print(f"Trade {i}:")
        print(f"  Buy:  {sell['buy_date'].strftime('%Y-%m-%d')} at ${sell['buy_price']:.2f}")
        print(f"  Sell: {sell['date'].strftime('%Y-%m-%d')} at ${sell['price']:.2f}")
        print(f"  Gain: {sell['gain']:.2%} ({days_held} days) - Reason: {sell['reason']}")
        print()
    
    # Summary statistics
    gains = [s['gain'] for s in sell_signals]
    profitable = [g for g in gains if g > 0]
    
    print("SUMMARY STATISTICS:")
    print("-" * 40)
    print(f"Total Trades: {len(sell_signals)}")
    print(f"Profitable Trades: {len(profitable)}")
    print(f"Win Rate: {len(profitable)/len(sell_signals):.1%}")
    print(f"Average Gain: {np.mean(gains):.2%}")
    print(f"Best Trade: {max(gains):.2%}")
    print(f"Worst Trade: {min(gains):.2%}")
    print(f"Total Return: {(np.prod([1 + g for g in gains]) - 1):.2%}")

def main():
    """Main function to generate ABBV 2015 visualization"""
    # Load ABBV data
    abbv_file = os.path.join('sd1', 'ABBV.csv')
    
    if not os.path.exists(abbv_file):
        print(f"Error: ABBV.csv not found at {abbv_file}")
        return
    
    print("Loading ABBV data...")
    df = load_and_prepare_data(abbv_file)
    
    print("Finding buy/sell signals for 2015...")
    # Using strategy parameters from results: 30% below 100MA, max 40 days
    year_data, buy_signals, sell_signals = find_buy_sell_signals(df, 2015, 0.7, 40)
    
    if len(year_data) == 0:
        print("No data found for 2015")
        return
    
    print("Creating visualization...")
    fig = create_visualization(year_data, buy_signals, sell_signals, "ABBV")
    
    # Print detailed trade analysis
    print_trade_details(buy_signals, sell_signals, "ABBV")

if __name__ == "__main__":
    main()