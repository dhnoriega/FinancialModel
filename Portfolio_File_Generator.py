import yfinance as yf
import pandas as pd
from datetime import datetime

def quick_test_ticker(ticker, start_date='2017-01-01'):
    """
    Quick test for a single ticker with minimal processing
    """
    try:
        print(f"Testing {ticker}...")
        
        # Download data
        data = yf.download(ticker, start=start_date, progress=False)
        
        print(f"Downloaded data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Data types:\n{data.dtypes}")
        print(f"First few rows:\n{data.head()}")
        
        if data.empty:
            print(f"❌ No data for {ticker}")
            return False
        
        # Reset index to get Date as column
        data = data.reset_index()
        
        # Handle MultiIndex columns if needed
        if isinstance(data.columns, pd.MultiIndex):
            print("MultiIndex detected, flattening...")
            data.columns = data.columns.droplevel(1)
        
        print(f"Columns after processing: {list(data.columns)}")
        
        # Check for required columns
        required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in data.columns]
        
        if missing:
            print(f"❌ Missing columns: {missing}")
            return False
        
        # Save file
        filename = f"{ticker}_features_optimized.csv"
        data[required].set_index('Date').to_csv(filename)
        
        # Verify
        test_read = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
        print(f"✅ Successfully created {filename} with {len(test_read)} rows")
        print(f"Date range: {test_read.index.min()} to {test_read.index.max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test with reliable tickers
test_tickers = ['AAPL', 'AMZN', 'GS', 'MA', 'NFLX', 'META', 
           'GOOGL', 'MSFT', 'GROY', 'EWO', 'SYM', 'AMIX', 
           'ESRT', 'INCY', 'BALL', 'MSS', 'FCAL', 'BME', 
           'LRGE', 'QSPT', 'BLV', 'SNAP', 'TSLA', 'V', 
           'USO', 'ALTG', 'QBTS', 'NLY', 'WTM', 'XSVM', 
           'AAT', 'XPAY', 'FEDM', 'CBUS']

print("Quick ticker test starting...")
print("=" * 50)

successful = []
for ticker in test_tickers:
    if quick_test_ticker(ticker):
        successful.append(ticker)
    print("-" * 30)

print(f"\nSUMMARY:")
print(f"Successful: {successful}")
print(f"Success rate: {len(successful)}/{len(test_tickers)} ({len(successful)/len(test_tickers)*100:.0f}%)")

if successful:
    print(f"\n✅ Ready to add to trading strategy: {successful}")
