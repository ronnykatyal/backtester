"""
Fixed Tick Data Processor with Robust Timestamp Handling
Handles mixed timestamp formats in your 7.1M tick dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def robust_timestamp_conversion(timestamp_series):
    """
    Handle mixed timestamp formats robustly.
    
    Your data has both:
    - 2025-02-01 00:00:02.520630 (with microseconds)
    - 2025-02-05 01:20:55 (without microseconds)
    """
    print("Converting timestamps with mixed formats...")
    
    # Try pandas mixed format first (fastest)
    try:
        converted = pd.to_datetime(timestamp_series, format='mixed', errors='coerce')
        invalid_count = converted.isna().sum()
        
        if invalid_count == 0:
            print("✅ All timestamps converted successfully with mixed format")
            return converted
        else:
            print(f"⚠️ {invalid_count} timestamps failed with mixed format, trying manual approach...")
    except Exception as e:
        print(f"Mixed format failed: {e}")
    
    # Manual approach for mixed formats
    print("Processing timestamps manually (this may take a moment for 7M records)...")
    
    def parse_single_timestamp(ts_str):
        """Parse individual timestamp with fallback formats"""
        if pd.isna(ts_str):
            return pd.NaT
        
        try:
            # Format 1: With microseconds
            if '.' in str(ts_str):
                return pd.to_datetime(ts_str, format='%Y-%m-%d %H:%M:%S.%f')
            # Format 2: Without microseconds  
            else:
                return pd.to_datetime(ts_str, format='%Y-%m-%d %H:%M:%S')
        except:
            # Last resort: let pandas infer
            try:
                return pd.to_datetime(ts_str)
            except:
                return pd.NaT
    
    # Process in chunks for memory efficiency
    chunk_size = 100000
    total_records = len(timestamp_series)
    converted_chunks = []
    
    for i in range(0, total_records, chunk_size):
        chunk = timestamp_series.iloc[i:i+chunk_size]
        converted_chunk = chunk.apply(parse_single_timestamp)
        converted_chunks.append(converted_chunk)
        
        # Progress update
        progress = min(i + chunk_size, total_records)
        print(f"  Processed {progress:,} / {total_records:,} timestamps ({progress/total_records*100:.1f}%)")
    
    # Combine all chunks
    converted = pd.concat(converted_chunks, ignore_index=True)
    invalid_count = converted.isna().sum()
    
    print(f"✅ Timestamp conversion complete: {len(converted) - invalid_count:,} valid, {invalid_count:,} invalid")
    
    return converted


def load_tick_data_robust(file_path):
    """
    Load tick data with robust error handling for your large dataset.
    """
    try:
        print(f"Loading large CSV file: {file_path}")
        print(f"File size: {Path(file_path).stat().st_size / (1024*1024):.1f} MB")
        
        # Load CSV with efficient settings for large files
        df = pd.read_csv(
            file_path,
            dtype={
                'product_symbol': 'category',  # Memory efficient for repeated values
                'price': 'float64',
                'size': 'float64', 
                'buyer_role': 'category'  # Memory efficient
            }
        )
        
        print(f"Loaded {len(df):,} rows")
        
        # Standardize column names
        column_mapping = {
            'product_symbol': 'symbol',
            'size': 'volume',
            'buyer_role': 'type'
        }
        df = df.rename(columns=column_mapping)
        
        # Convert timestamp with robust handling
        df['timestamp'] = robust_timestamp_conversion(df['timestamp'])
        
        # Convert price and volume to numeric (should already be correct)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Clean type field
        df['type'] = df['type'].astype(str).str.lower().str.strip()
        
        # Remove any invalid rows
        original_length = len(df)
        df = df.dropna(subset=['price', 'volume', 'timestamp'])
        removed_count = original_length - len(df)
        
        if removed_count > 0:
            print(f"Removed {removed_count:,} invalid rows")
        
        # Sort by timestamp for time series processing
        print("Sorting by timestamp...")
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Data loaded successfully: {len(df):,} valid records")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading tick data: {e}")
        raise


def resample_ticks_to_ohlcv_efficient(tick_df, timeframe='1H'):
    """
    Efficiently convert large tick dataset to OHLCV format.
    """
    print(f"Resampling {len(tick_df):,} ticks to {timeframe} bars...")
    
    # Set timestamp as index for resampling
    tick_df = tick_df.set_index('timestamp')
    
    # Efficient resampling for large datasets
    ohlcv = tick_df.groupby(pd.Grouper(freq=timeframe)).agg({
        'price': ['first', 'max', 'min', 'last', 'count'],
        'volume': 'sum'
    })
    
    # Flatten column names
    ohlcv.columns = ['open', 'high', 'low', 'close', 'tick_count', 'volume']
    
    # Remove periods with no data
    ohlcv = ohlcv.dropna()
    
    print(f"Created {len(ohlcv)} {timeframe} bars")
    
    return ohlcv


def add_tick_features_efficient(tick_df, ohlcv_df, lookback_minutes=60):
    """
    Add tick-based features efficiently for large datasets.
    """
    print("Adding tick-based features (this may take a few minutes for large datasets)...")
    
    enhanced_df = ohlcv_df.copy()
    
    # Initialize feature columns
    enhanced_df['order_flow_ratio'] = 0.0
    enhanced_df['volume_imbalance'] = 0.0
    enhanced_df['price_momentum'] = 0.0
    enhanced_df['taker_volume_ratio'] = 0.0
    enhanced_df['large_order_count'] = 0
    
    total_bars = len(enhanced_df)
    processed = 0
    
    # Process each OHLCV bar
    for idx, row in enhanced_df.iterrows():
        # Get tick data for this time period
        end_time = idx
        start_time = end_time - pd.Timedelta(minutes=lookback_minutes)
        
        # Filter tick data efficiently
        mask = (tick_df['timestamp'] >= start_time) & (tick_df['timestamp'] <= end_time)
        period_ticks = tick_df[mask]
        
        if len(period_ticks) > 0:
            # Calculate features for this period
            features = calculate_tick_features_fast(period_ticks)
            
            # Update features
            for feature_name, feature_value in features.items():
                if feature_name in enhanced_df.columns:
                    enhanced_df.loc[idx, feature_name] = feature_value
        
        # Progress update
        processed += 1
        if processed % 50 == 0 or processed == total_bars:
            print(f"  Processed {processed}/{total_bars} bars ({processed/total_bars*100:.1f}%)")
    
    print("✅ Tick features added successfully")
    return enhanced_df


def calculate_tick_features_fast(tick_data):
    """
    Fast calculation of tick features using vectorized operations.
    """
    if len(tick_data) == 0:
        return {
            'order_flow_ratio': 0.5,
            'volume_imbalance': 0.0,
            'price_momentum': 0.0,
            'taker_volume_ratio': 0.0,
            'large_order_count': 0
        }
    
    # Vectorized calculations
    price_changes = tick_data['price'].diff()
    
    # Order flow analysis
    up_mask = price_changes > 0
    down_mask = price_changes < 0
    
    up_volume = tick_data[up_mask]['volume'].sum()
    down_volume = tick_data[down_mask]['volume'].sum()
    total_directional_volume = up_volume + down_volume
    
    order_flow_ratio = up_volume / total_directional_volume if total_directional_volume > 0 else 0.5
    
    # Volume imbalance
    total_volume = tick_data['volume'].sum()
    volume_imbalance = (up_volume - down_volume) / total_volume if total_volume > 0 else 0.0
    
    # Price momentum
    if len(tick_data) > 1:
        price_momentum = (tick_data['price'].iloc[-1] / tick_data['price'].iloc[0] - 1) * 100
    else:
        price_momentum = 0.0
    
    # Taker volume ratio
    taker_volume = tick_data[tick_data['type'] == 'taker']['volume'].sum()
    taker_volume_ratio = taker_volume / total_volume if total_volume > 0 else 0.0
    
    # Large order count (top 10% by volume)
    if len(tick_data) >= 10:
        volume_threshold = tick_data['volume'].quantile(0.9)
        large_order_count = (tick_data['volume'] >= volume_threshold).sum()
    else:
        large_order_count = 0
    
    return {
        'order_flow_ratio': round(order_flow_ratio, 3),
        'volume_imbalance': round(volume_imbalance, 3),
        'price_momentum': round(price_momentum, 4),
        'taker_volume_ratio': round(taker_volume_ratio, 3),
        'large_order_count': large_order_count
    }


def create_enhanced_dataset_robust(tick_file_path, timeframe='1H', output_file=None):
    """
    Main function optimized for large tick datasets.
    """
    print(f"🚀 Processing large tick dataset: {tick_file_path}")
    
    # Load tick data with robust handling
    tick_df = load_tick_data_robust(tick_file_path)
    
    # Convert to OHLCV
    ohlcv_df = resample_ticks_to_ohlcv_efficient(tick_df, timeframe)
    
    # Add tick-based features
    enhanced_df = add_tick_features_efficient(tick_df, ohlcv_df)
    
    # Save results
    if output_file:
        # Ensure directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        enhanced_df.to_csv(output_file)
        print(f"✅ Enhanced dataset saved to: {output_file}")
    
    print(f"🎯 Processing complete! Created {len(enhanced_df)} enhanced bars from {len(tick_df):,} ticks")
    
    return enhanced_df


def main():
    """Process your massive tick dataset"""
    tick_file = "futures_BTCUSD_2025_02_trades.csv"
    
    if Path(tick_file).exists():
        print("🚀 Starting robust processing of your 7.1M tick dataset...")
        print("⏱️  This will take 3-5 minutes for optimal feature extraction...")
        
        try:
            # Process with robust error handling
            enhanced_data = create_enhanced_dataset_robust(
                tick_file_path=tick_file,
                timeframe='1H',
                output_file='data/BTCUSDT_1h_enhanced.csv'
            )
            
            # Show results
            print("\n📊 PROCESSING RESULTS:")
            print("=" * 50)
            print(f"✅ Created: {len(enhanced_data)} hourly bars")
            print(f"✅ Date range: {enhanced_data.index.min()} to {enhanced_data.index.max()}")
            print(f"✅ Price range: ${enhanced_data['close'].min():,.0f} - ${enhanced_data['close'].max():,.0f}")
            
            print("\n🎯 TICK FEATURES SUMMARY:")
            tick_features = ['order_flow_ratio', 'volume_imbalance', 'taker_volume_ratio', 'large_order_count']
            print(enhanced_data[tick_features].describe().round(3))
            
            print("\n🚀 READY FOR ENHANCED BACKTESTING!")
            print("Next step: Run your enhanced EMA strategy!")
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            print("This might be due to memory limitations with such a large dataset.")
            print("Consider processing smaller chunks or increasing system memory.")
    
    else:
        print(f"❌ File not found: {tick_file}")


if __name__ == "__main__":
    main()