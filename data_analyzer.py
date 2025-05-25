import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


class OHLCVDataAnalyzer:
    """Advanced OHLCV data analyzer with financial trading knowledge"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare and clean the dataset"""
        # Convert timestamp if needed
        if 'timestamp' in self.df.columns:
            # Handle Excel date serial numbers
            if self.df['timestamp'].dtype in ['int64', 'float64']:
                # Convert Excel serial date to datetime
                self.df['timestamp'] = pd.to_datetime('1900-01-01') + pd.to_timedelta(self.df['timestamp'] - 2, unit='D')
            else:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Parse support and resistance arrays
        if 'Support' in self.df.columns:
            self.df['support_levels'] = self.df['Support'].apply(self.parse_price_levels)
            self.df['support_min'] = self.df['support_levels'].apply(lambda x: min(x) if x else None)
            self.df['support_max'] = self.df['support_levels'].apply(lambda x: max(x) if x else None)
        
        if 'Resistance' in self.df.columns:
            self.df['resistance_levels'] = self.df['Resistance'].apply(self.parse_price_levels)
            self.df['resistance_min'] = self.df['resistance_levels'].apply(lambda x: min(x) if x else None)
            self.df['resistance_max'] = self.df['resistance_levels'].apply(lambda x: max(x) if x else None)
        
        # Calculate additional technical indicators
        self.calculate_technical_indicators()
    
    def parse_price_levels(self, level_str) -> List[float]:
        """Parse support/resistance level strings into lists of floats"""
        if pd.isna(level_str) or level_str == '':
            return []
        try:
            # Handle string representation of lists
            if isinstance(level_str, str):
                level_str = level_str.strip()
                if level_str.startswith('[') and level_str.endswith(']'):
                    return json.loads(level_str)
            elif isinstance(level_str, list):
                return level_str
            return []
        except:
            return []
    
    def calculate_technical_indicators(self):
        """Calculate various technical indicators"""
        # Price changes
        self.df['price_change'] = self.df['close'] - self.df['open']
        self.df['price_change_pct'] = (self.df['price_change'] / self.df['open']) * 100
        
        # Bullish/Bearish classification
        self.df['is_bullish'] = self.df['close'] > self.df['open']
        self.df['is_bearish'] = self.df['close'] < self.df['open']
        
        # Price range and volatility
        self.df['range'] = self.df['high'] - self.df['low']
        self.df['range_pct'] = (self.df['range'] / self.df['open']) * 100
        
        # Moving averages (if enough data)
        if len(self.df) >= 20:
            self.df['sma_20'] = self.df['close'].rolling(window=20).mean()
        if len(self.df) >= 50:
            self.df['sma_50'] = self.df['close'].rolling(window=50).mean()
        
        # Volume analysis
        if 'volume' in self.df.columns:
            self.df['volume_sma'] = self.df['volume'].rolling(window=20).mean()
            self.df['high_volume'] = self.df['volume'] > self.df['volume_sma'] 