import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import os
from collections import Counter

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.tools import Tool
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function

# Set page config
st.set_page_config(
    page_title="OHLCV Financial Data Analyst",
    page_icon="📈",
    layout="wide"
)

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

class OHLCVChatbotTools:
    """Advanced custom tools for comprehensive OHLCV chatbot analysis"""
    
    def __init__(self, analyzer: OHLCVDataAnalyzer):
        self.analyzer = analyzer
        self.df = analyzer.df
        self.prepare_advanced_indicators()
    
    def prepare_advanced_indicators(self):
        """Prepare advanced technical indicators and metrics"""
        # Price gaps analysis
        if len(self.df) > 1:
            self.df['price_gap'] = self.df['open'] - self.df['close'].shift(1)
            self.df['gap_pct'] = (self.df['price_gap'] / self.df['close'].shift(1)) * 100
        
        # Support/Resistance distances
        self.df['support_distance'] = self.df.apply(
            lambda row: abs(row['close'] - row['support_min']) if pd.notna(row['support_min']) else None, axis=1
        )
        self.df['resistance_distance'] = self.df.apply(
            lambda row: abs(row['resistance_max'] - row['close']) if pd.notna(row['resistance_max']) else None, axis=1
        )
        
        # Support/Resistance range
        self.df['sr_range'] = self.df['resistance_max'] - self.df['support_min']
        
        # Breakout detection
        self.df['support_break'] = self.df['low'] < self.df['support_min']
        self.df['resistance_break'] = self.df['high'] > self.df['resistance_max']
        
        # Signal analysis preparation
        self.df['has_signal'] = self.df['direction'].notna()
        self.df['signal_accuracy'] = None  # Will be calculated dynamically
        
        # Volume analysis
        if 'volume' in self.df.columns:
            self.df['volume_ma'] = self.df['volume'].rolling(window=20, min_periods=1).mean()
            self.df['high_volume'] = self.df['volume'] > self.df['volume_ma']
            self.df['volume_ratio'] = self.df['volume'] / self.df['volume_ma']
    
    def get_basic_stats(self, query: str = "") -> str:
        """Get comprehensive statistics about the OHLCV dataset"""
        # Basic dataset info
        total_records = len(self.df)
        signal_records = self.df['direction'].notna().sum()
        signal_coverage = (signal_records / total_records) * 100
        
        # Date range analysis
        date_info = "N/A"
        if 'timestamp' in self.df.columns:
            start_date = self.df['timestamp'].min()
            end_date = self.df['timestamp'].max()
            date_range = (end_date - start_date).days
            date_info = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({date_range} days)"
        
        # Price statistics
        price_stats = {
            "lowest_low": float(self.df['low'].min()),
            "highest_high": float(self.df['high'].max()),
            "avg_close": float(self.df['close'].mean()),
            "price_range_total": float(self.df['high'].max() - self.df['low'].min())
        }
        
        # Volume statistics
        volume_stats = {}
        if 'volume' in self.df.columns:
            volume_stats = {
                "total_volume": float(self.df['volume'].sum()),
                "avg_volume": float(self.df['volume'].mean()),
                "max_volume": float(self.df['volume'].max())
            }
        
        # Signal distribution
        signal_dist = self.df['direction'].value_counts().to_dict()
        
        return f"""📊 **COMPREHENSIVE DATASET OVERVIEW**

**📈 Dataset Fundamentals:**
- Total Trading Periods: {total_records:,}
- Date Range: {date_info}
- Signal Coverage: {signal_records}/{total_records} ({signal_coverage:.1f}%)

**💰 Price Analysis:**
- Trading Range: ${price_stats['lowest_low']:.2f} - ${price_stats['highest_high']:.2f}
- Total Price Range: ${price_stats['price_range_total']:.2f}
- Average Close Price: ${price_stats['avg_close']:.2f}

**📊 Volume Statistics:**
- Total Volume Traded: {volume_stats.get('total_volume', 0):,.0f}
- Average Daily Volume: {volume_stats.get('avg_volume', 0):,.0f}
- Maximum Volume Day: {volume_stats.get('max_volume', 0):,.0f}

**🎯 Trading Signals Distribution:**
- LONG Signals: {signal_dist.get('LONG', 0)}
- SHORT Signals: {signal_dist.get('SHORT', 0)}
- No Signal Periods: {total_records - signal_records}

**🎪 Support & Resistance Coverage:**
- Support Levels Available: {(self.df['support_levels'].apply(len) > 0).sum()}/{total_records}
- Resistance Levels Available: {(self.df['resistance_levels'].apply(len) > 0).sum()}/{total_records}
- Average S/R Range: ${self.df['sr_range'].mean():.2f}"""

    def analyze_signal_performance(self, query: str = "") -> str:
        """Comprehensive analysis of trading signal performance and accuracy"""
        if self.df['direction'].isna().all():
            return "No trading signals found in the dataset."
        
        # Basic signal stats
        long_signals = self.df[self.df['direction'] == 'LONG']
        short_signals = self.df[self.df['direction'] == 'SHORT']
        no_signal = self.df[self.df['direction'].isna()]
        
        # Calculate future returns for signal accuracy (1-period ahead)
        self.df['future_return'] = self.df['close'].shift(-1) / self.df['close'] - 1
        
        # Signal accuracy analysis
        long_accuracy = ((long_signals['future_return'] > 0).sum() / len(long_signals) * 100) if len(long_signals) > 0 else 0
        short_accuracy = ((short_signals['future_return'] < 0).sum() / len(short_signals) * 100) if len(short_signals) > 0 else 0
        
        # Average returns following signals
        long_avg_return = long_signals['future_return'].mean() * 100 if len(long_signals) > 0 else 0
        short_avg_return = short_signals['future_return'].mean() * 100 if len(short_signals) > 0 else 0
        no_signal_return = no_signal['future_return'].mean() * 100 if len(no_signal) > 0 else 0
        
        # Signal timing analysis
        signal_periods = self.df[self.df['direction'].notna()]
        
        # Volume confirmation
        long_with_high_vol = long_signals[long_signals['high_volume']].shape[0] if 'high_volume' in long_signals.columns else 0
        short_with_high_vol = short_signals[short_signals['high_volume']].shape[0] if 'high_volume' in short_signals.columns else 0
        
        # Support/Resistance context
        long_near_support = (long_signals['support_distance'] < long_signals['sr_range'] * 0.1).sum() if len(long_signals) > 0 else 0
        short_near_resistance = (short_signals['resistance_distance'] < short_signals['sr_range'] * 0.1).sum() if len(short_signals) > 0 else 0
        
        return f"""🎯 **TRADING SIGNAL PERFORMANCE ANALYSIS**

**📊 Signal Distribution:**
- LONG Signals: {len(long_signals)} ({len(long_signals)/len(self.df)*100:.1f}% of dataset)
- SHORT Signals: {len(short_signals)} ({len(short_signals)/len(self.df)*100:.1f}% of dataset)
- No Signal Periods: {len(no_signal)} ({len(no_signal)/len(self.df)*100:.1f}% of dataset)

**🎯 Signal Accuracy (Next Period):**
- LONG Signal Success Rate: {long_accuracy:.1f}%
- SHORT Signal Success Rate: {short_accuracy:.1f}%
- Overall Signal Accuracy: {(long_accuracy + short_accuracy)/2:.1f}%

**💰 Average Returns Following Signals:**
- LONG Signals: {long_avg_return:+.2f}%
- SHORT Signals: {short_avg_return:+.2f}%
- No Signal Periods: {no_signal_return:+.2f}%

**📊 Volume Confirmation:**
- LONG signals with high volume: {long_with_high_vol}/{len(long_signals)} ({long_with_high_vol/len(long_signals)*100 if len(long_signals) > 0 else 0:.1f}%)
- SHORT signals with high volume: {short_with_high_vol}/{len(short_signals)} ({short_with_high_vol/len(short_signals)*100 if len(short_signals) > 0 else 0:.1f}%)

**🎪 Support/Resistance Context:**
- LONG signals near support: {long_near_support}/{len(long_signals)} ({long_near_support/len(long_signals)*100 if len(long_signals) > 0 else 0:.1f}%)
- SHORT signals near resistance: {short_near_resistance}/{len(short_signals)} ({short_near_resistance/len(short_signals)*100 if len(short_signals) > 0 else 0:.1f}%)

**💡 Key Insights:**
- Signal vs No-Signal Performance Difference: {((long_avg_return + short_avg_return)/2 - no_signal_return):+.2f}%
- Best Performing Signal Type: {'LONG' if long_avg_return > abs(short_avg_return) else 'SHORT'}
- Volume Confirmation Impact: {'Positive' if (long_with_high_vol/len(long_signals) if len(long_signals) > 0 else 0) > 0.5 else 'Mixed'}"""

    def analyze_support_resistance_advanced(self, query: str = "") -> str:
        """Advanced support and resistance level analysis"""
        if 'support_levels' not in self.df.columns:
            return "Support and resistance data not available in the dataset."
        
        # Breakout analysis
        support_breaks = self.df['support_break'].sum()
        resistance_breaks = self.df['resistance_break'].sum()
        
        # Support/Resistance effectiveness
        support_holds = len(self.df) - support_breaks
        resistance_holds = len(self.df) - resistance_breaks
        
        # Average distances and ranges
        avg_support_distance = self.df['support_distance'].mean()
        avg_resistance_distance = self.df['resistance_distance'].mean()
        avg_sr_range = self.df['sr_range'].mean()
        
        # Level frequency analysis
        all_support_levels = []
        all_resistance_levels = []
        
        for levels in self.df['support_levels'].dropna():
            if isinstance(levels, list):
                all_support_levels.extend(levels)
        
        for levels in self.df['resistance_levels'].dropna():
            if isinstance(levels, list):
                all_resistance_levels.extend(levels)
        
        # Most common levels (rounded to nearest 5)
        support_rounded = [round(level/5)*5 for level in all_support_levels]
        resistance_rounded = [round(level/5)*5 for level in all_resistance_levels]
        
        top_support = Counter(support_rounded).most_common(5)
        top_resistance = Counter(resistance_rounded).most_common(5)
        
        # Breakout magnitude analysis
        breakout_data = self.df[self.df['resistance_break']]
        breakdown_data = self.df[self.df['support_break']]
        
        avg_breakout_magnitude = (breakout_data['high'] - breakout_data['resistance_max']).mean() if len(breakout_data) > 0 else 0
        avg_breakdown_magnitude = (breakdown_data['support_min'] - breakdown_data['low']).mean() if len(breakdown_data) > 0 else 0
        
        # Volume during breaks
        breakout_volume = breakout_data['volume'].mean() if len(breakout_data) > 0 else 0
        breakdown_volume = breakdown_data['volume'].mean() if len(breakdown_data) > 0 else 0
        normal_volume = self.df['volume'].mean()
        
        return f"""🎪 **ADVANCED SUPPORT & RESISTANCE ANALYSIS**

**🛡️ Level Effectiveness:**
- Support Levels Held: {support_holds}/{len(self.df)} ({support_holds/len(self.df)*100:.1f}%)
- Resistance Levels Held: {resistance_holds}/{len(self.df)} ({resistance_holds/len(self.df)*100:.1f}%)
- Support Breaks: {support_breaks} ({support_breaks/len(self.df)*100:.1f}%)
- Resistance Breaks: {resistance_breaks} ({resistance_breaks/len(self.df)*100:.1f}%)

**📏 Distance & Range Analysis:**
- Average Distance to Support: ${avg_support_distance:.2f}
- Average Distance to Resistance: ${avg_resistance_distance:.2f}
- Average Support-Resistance Range: ${avg_sr_range:.2f}

**🎯 Most Frequently Tested Levels:**
Top Support Levels: {', '.join([f'${level}({count}x)' for level, count in top_support[:3]])}
Top Resistance Levels: {', '.join([f'${level}({count}x)' for level, count in top_resistance[:3]])}

**💥 Breakout Analysis:**
- Average Breakout Magnitude: ${avg_breakout_magnitude:.2f}
- Average Breakdown Magnitude: ${avg_breakdown_magnitude:.2f}
- Breakout Success Rate: {resistance_breaks/(resistance_breaks + resistance_holds)*100:.1f}%
- Breakdown Rate: {support_breaks/(support_breaks + support_holds)*100:.1f}%

**📊 Volume During Breaks:**
- Normal Volume: {normal_volume:,.0f}
- Breakout Volume: {breakout_volume:,.0f} ({breakout_volume/normal_volume:.1f}x normal)
- Breakdown Volume: {breakdown_volume:,.0f} ({breakdown_volume/normal_volume:.1f}x normal)

**💡 Key Insights:**
- Stronger Level Type: {'Support' if support_holds/len(self.df) > resistance_holds/len(self.df) else 'Resistance'}
- Volume Confirmation: {'Strong' if max(breakout_volume, breakdown_volume) > normal_volume*1.5 else 'Weak'}
- Range Trading Opportunity: {'High' if avg_sr_range > avg_support_distance + avg_resistance_distance else 'Low'}"""

    def analyze_volume_price_relationship(self, query: str = "") -> str:
        """Comprehensive volume-price relationship analysis"""
        if 'volume' not in self.df.columns:
            return "Volume data not available in the dataset."
        
        # Volume-price correlation
        price_change_corr = self.df['price_change_pct'].corr(self.df['volume'])
        volatility_corr = self.df['range_pct'].corr(self.df['volume'])
        
        # Volume during different market conditions
        bullish_volume = self.df[self.df['is_bullish']]['volume'].mean()
        bearish_volume = self.df[self.df['is_bearish']]['volume'].mean()
        
        # Volume during breakouts
        breakout_periods = self.df[self.df['resistance_break'] | self.df['support_break']]
        normal_periods = self.df[~(self.df['resistance_break'] | self.df['support_break'])]
        
        breakout_avg_volume = breakout_periods['volume'].mean() if len(breakout_periods) > 0 else 0
        normal_avg_volume = normal_periods['volume'].mean() if len(normal_periods) > 0 else 0
        
        # High volume analysis
        high_vol_periods = self.df[self.df['high_volume']]
        high_vol_returns = high_vol_periods['price_change_pct'].mean()
        normal_vol_returns = self.df[~self.df['high_volume']]['price_change_pct'].mean()
        
        # Volume during signals
        signal_volume_analysis = ""
        if self.df['direction'].notna().any():
            long_signal_volume = self.df[self.df['direction'] == 'LONG']['volume'].mean()
            short_signal_volume = self.df[self.df['direction'] == 'SHORT']['volume'].mean()
            no_signal_volume = self.df[self.df['direction'].isna()]['volume'].mean()
            
            signal_volume_analysis = f"""
**📊 Volume During Trading Signals:**
- LONG Signal Volume: {long_signal_volume:,.0f} ({long_signal_volume/self.df['volume'].mean():.1f}x average)
- SHORT Signal Volume: {short_signal_volume:,.0f} ({short_signal_volume/self.df['volume'].mean():.1f}x average)
- No Signal Volume: {no_signal_volume:,.0f} ({no_signal_volume/self.df['volume'].mean():.1f}x average)"""
        
        # Accumulation/Distribution analysis
        accumulation_periods = self.df[(self.df['volume'] > self.df['volume'].quantile(0.75)) & 
                                     (self.df['range_pct'] < self.df['range_pct'].quantile(0.25))]
        distribution_periods = self.df[(self.df['volume'] > self.df['volume'].quantile(0.75)) & 
                                     (self.df['range_pct'] > self.df['range_pct'].quantile(0.75))]
        
        return f"""📊 **VOLUME-PRICE RELATIONSHIP ANALYSIS**

**🔗 Correlation Analysis:**
- Volume vs Price Change: {price_change_corr:.3f}
- Volume vs Volatility: {volatility_corr:.3f}
- Interpretation: {'Strong positive' if abs(volatility_corr) > 0.5 else 'Moderate' if abs(volatility_corr) > 0.3 else 'Weak'} volume-volatility relationship

**📈 Volume During Market Conditions:**
- Bullish Days Average Volume: {bullish_volume:,.0f}
- Bearish Days Average Volume: {bearish_volume:,.0f}
- Volume Bias: {'Bullish' if bullish_volume > bearish_volume else 'Bearish'} days show {abs(bullish_volume-bearish_volume)/min(bullish_volume,bearish_volume)*100:.1f}% higher volume

**💥 Breakout Volume Analysis:**
- Normal Period Volume: {normal_avg_volume:,.0f}
- Breakout Period Volume: {breakout_avg_volume:,.0f}
- Breakout Volume Multiplier: {breakout_avg_volume/normal_avg_volume:.1f}x
- Breakout Confirmation: {'Strong' if breakout_avg_volume > normal_avg_volume*1.5 else 'Weak'}

**⚡ High Volume Impact:**
- High Volume Periods: {len(high_vol_periods)} ({len(high_vol_periods)/len(self.df)*100:.1f}%)
- Returns During High Volume: {high_vol_returns:+.2f}%
- Returns During Normal Volume: {normal_vol_returns:+.2f}%
- High Volume Advantage: {high_vol_returns - normal_vol_returns:+.2f}%

{signal_volume_analysis}

**🏗️ Accumulation/Distribution Analysis:**
- Accumulation Periods (High Vol, Low Range): {len(accumulation_periods)} ({len(accumulation_periods)/len(self.df)*100:.1f}%)
- Distribution Periods (High Vol, High Range): {len(distribution_periods)} ({len(distribution_periods)/len(self.df)*100:.1f}%)

**💡 Key Volume Insights:**
- Volume Confirmation Quality: {'Excellent' if breakout_avg_volume > normal_avg_volume*2 else 'Good' if breakout_avg_volume > normal_avg_volume*1.5 else 'Poor'}
- Best Volume Strategy: {'Trade with high volume' if high_vol_returns > normal_vol_returns else 'Volume not significant predictor'}
- Market Participation: {'Increasing on moves' if volatility_corr > 0.3 else 'Steady regardless of movement'}"""

    def analyze_bullish_bearish_days(self, query: str = "") -> str:
        """Analyze bullish and bearish days in the dataset"""
        bullish_days = self.df['is_bullish'].sum()
        bearish_days = self.df['is_bearish'].sum()
        neutral_days = len(self.df) - bullish_days - bearish_days
        
        # Year-wise analysis if timestamp is available
        yearly_analysis = ""
        if 'timestamp' in self.df.columns:
            self.df['year'] = self.df['timestamp'].dt.year
            yearly_stats = self.df.groupby('year').agg({
                'is_bullish': 'sum',
                'is_bearish': 'sum',
                'price_change_pct': 'mean'
            }).round(2)
            
            yearly_analysis = "\n\nYear-wise Breakdown:\n"
            for year, stats in yearly_stats.iterrows():
                yearly_analysis += f"- {year}: {stats['is_bullish']} bullish, {stats['is_bearish']} bearish (avg change: {stats['price_change_pct']:.2f}%)\n"
        
        return f"""Bullish vs Bearish Analysis:
- Bullish Days (Close > Open): {bullish_days} ({bullish_days/len(self.df)*100:.1f}%)
- Bearish Days (Close < Open): {bearish_days} ({bearish_days/len(self.df)*100:.1f}%)
- Neutral Days (Close = Open): {neutral_days} ({neutral_days/len(self.df)*100:.1f}%)

Average Price Changes:
- Bullish days avg change: {self.df[self.df['is_bullish']]['price_change_pct'].mean():.2f}%
- Bearish days avg change: {self.df[self.df['is_bearish']]['price_change_pct'].mean():.2f}%
{yearly_analysis}"""

    def analyze_volatility(self, query: str = "") -> str:
        """Analyze price volatility and trading ranges"""
        # Daily volatility metrics
        daily_volatility = self.df['range_pct'].mean()
        max_volatility = self.df['range_pct'].max()
        min_volatility = self.df['range_pct'].min()
        
        # Price gap analysis
        gaps_up = gaps_down = avg_gap = 0
        if len(self.df) > 1 and 'gap_pct' in self.df.columns:
            gaps_up = (self.df['gap_pct'] > 0).sum()
            gaps_down = (self.df['gap_pct'] < 0).sum()
            avg_gap = self.df['gap_pct'].mean()
        
        # Volume-volatility correlation
        volume_corr = ""
        if 'volume' in self.df.columns:
            corr = self.df['range_pct'].corr(self.df['volume'])
            volume_corr = f"\nVolume-Volatility Correlation: {corr:.3f}"
        
        return f"""Volatility Analysis:
        
Daily Price Ranges:
- Average daily range: {daily_volatility:.2f}%
- Maximum daily range: {max_volatility:.2f}%
- Minimum daily range: {min_volatility:.2f}%

Price Gaps (Open vs Previous Close):
- Gap ups: {gaps_up}
- Gap downs: {gaps_down}
- Average gap: {avg_gap:.2f}%

Volatility Distribution:
- High volatility days (>5%): {(self.df['range_pct'] > 5).sum()}
- Medium volatility days (2-5%): {((self.df['range_pct'] >= 2) & (self.df['range_pct'] <= 5)).sum()}
- Low volatility days (<2%): {(self.df['range_pct'] < 2).sum()}
{volume_corr}"""

    def analyze_market_regimes(self, query: str = "") -> str:
        """Analyze different market regimes and patterns"""
        # Trend analysis using moving averages
        if 'sma_20' in self.df.columns and len(self.df) >= 20:
            self.df['trend_direction'] = np.where(self.df['close'] > self.df['sma_20'], 'Uptrend', 
                                                np.where(self.df['close'] < self.df['sma_20'], 'Downtrend', 'Sideways'))
            
            trend_counts = self.df['trend_direction'].value_counts()
            
            # Trend strength
            uptrend_periods = self.df[self.df['trend_direction'] == 'Uptrend']
            downtrend_periods = self.df[self.df['trend_direction'] == 'Downtrend']
            sideways_periods = self.df[self.df['trend_direction'] == 'Sideways']
            
            uptrend_performance = uptrend_periods['price_change_pct'].mean() if len(uptrend_periods) > 0 else 0
            downtrend_performance = downtrend_periods['price_change_pct'].mean() if len(downtrend_periods) > 0 else 0
            sideways_performance = sideways_periods['price_change_pct'].mean() if len(sideways_periods) > 0 else 0
        else:
            trend_counts = pd.Series({'Uptrend': 0, 'Downtrend': 0, 'Sideways': 0})
            uptrend_performance = downtrend_performance = sideways_performance = 0
        
        # Volatility regimes
        vol_median = self.df['range_pct'].median()
        self.df['volatility_regime'] = np.where(self.df['range_pct'] > vol_median * 1.5, 'High Vol',
                                              np.where(self.df['range_pct'] < vol_median * 0.5, 'Low Vol', 'Normal Vol'))
        
        vol_regime_counts = self.df['volatility_regime'].value_counts()
        
        # Signal clustering analysis
        avg_cluster_size = max_cluster_size = 0
        if self.df['direction'].notna().any():
            signal_periods = self.df[self.df['direction'].notna()].copy()
            if len(signal_periods) > 0:
                signal_periods['signal_group'] = (signal_periods['direction'] != signal_periods['direction'].shift()).cumsum()
                signal_clusters = signal_periods.groupby('signal_group').agg({
                    'direction': ['first', 'count'],
                    'price_change_pct': 'sum'
                }).round(2)
                
                avg_cluster_size = signal_clusters[('direction', 'count')].mean()
                max_cluster_size = signal_clusters[('direction', 'count')].max()
        
        # Market regime transitions
        regime_changes = 0
        if 'trend_direction' in self.df.columns:
            regime_changes = (self.df['trend_direction'] != self.df['trend_direction'].shift()).sum()
        
        # Consolidation periods (tight S/R ranges)
        tight_ranges = self.df[self.df['sr_range'] < self.df['sr_range'].quantile(0.25)]
        wide_ranges = self.df[self.df['sr_range'] > self.df['sr_range'].quantile(0.75)]
        
        return f"""🌊 **MARKET REGIME & PATTERN ANALYSIS**

**📈 Trend Classification (20-day MA):**
- Uptrend Periods: {trend_counts.get('Uptrend', 0)} ({trend_counts.get('Uptrend', 0)/len(self.df)*100:.1f}%)
- Downtrend Periods: {trend_counts.get('Downtrend', 0)} ({trend_counts.get('Downtrend', 0)/len(self.df)*100:.1f}%)
- Sideways Periods: {trend_counts.get('Sideways', 0)} ({trend_counts.get('Sideways', 0)/len(self.df)*100:.1f}%)

**💰 Regime Performance:**
- Uptrend Average Return: {uptrend_performance:+.2f}%
- Downtrend Average Return: {downtrend_performance:+.2f}%
- Sideways Average Return: {sideways_performance:+.2f}%

**⚡ Volatility Regimes:**
- High Volatility Periods: {vol_regime_counts.get('High Vol', 0)} ({vol_regime_counts.get('High Vol', 0)/len(self.df)*100:.1f}%)
- Normal Volatility Periods: {vol_regime_counts.get('Normal Vol', 0)} ({vol_regime_counts.get('Normal Vol', 0)/len(self.df)*100:.1f}%)
- Low Volatility Periods: {vol_regime_counts.get('Low Vol', 0)} ({vol_regime_counts.get('Low Vol', 0)/len(self.df)*100:.1f}%)

**🎯 Signal Clustering:**
- Average Signal Cluster Size: {avg_cluster_size:.1f} consecutive periods
- Maximum Signal Cluster: {max_cluster_size} consecutive periods

**🔄 Market Transitions:**
- Trend Regime Changes: {regime_changes} transitions
- Average Regime Duration: {len(self.df)/max(regime_changes, 1):.1f} periods

**📦 Range Analysis:**
- Tight Range Periods: {len(tight_ranges)} ({len(tight_ranges)/len(self.df)*100:.1f}%) - Good for range trading
- Wide Range Periods: {len(wide_ranges)} ({len(wide_ranges)/len(self.df)*100:.1f}%) - Trending/volatile markets

**💡 Regime Insights:**
- Dominant Market State: {trend_counts.idxmax() if len(trend_counts) > 0 else 'Analysis pending'}
- Best Performing Regime: {'Uptrend' if uptrend_performance > max(downtrend_performance, sideways_performance) else 'Downtrend' if downtrend_performance > sideways_performance else 'Sideways'}
- Volatility Pattern: {'Clustered' if vol_regime_counts.get('High Vol', 0) + vol_regime_counts.get('Low Vol', 0) > vol_regime_counts.get('Normal Vol', 0) else 'Consistent'}
- Signal Consistency: {'Clustered' if avg_cluster_size > 2 else 'Scattered'}"""

    def analyze_risk_and_drawdowns(self, query: str = "") -> str:
        """Comprehensive risk analysis including drawdowns and volatility metrics"""
        # Calculate cumulative returns
        self.df['cumulative_return'] = (1 + self.df['price_change_pct']/100).cumprod()
        
        # Drawdown calculation
        self.df['rolling_max'] = self.df['cumulative_return'].expanding().max()
        self.df['drawdown'] = (self.df['cumulative_return'] - self.df['rolling_max']) / self.df['rolling_max'] * 100
        
        # Maximum drawdown
        max_drawdown = self.df['drawdown'].min()
        max_dd_start = self.df['drawdown'].idxmin()
        
        # Drawdown duration
        in_drawdown = self.df['drawdown'] < -1  # More than 1% drawdown
        drawdown_periods = in_drawdown.sum()
        
        # Risk metrics
        daily_volatility = self.df['price_change_pct'].std()
        annualized_volatility = daily_volatility * np.sqrt(252)  # Assuming daily data
        
        # Value at Risk (VaR) - 5% and 1%
        var_5 = self.df['price_change_pct'].quantile(0.05)
        var_1 = self.df['price_change_pct'].quantile(0.01)
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        excess_returns = self.df['price_change_pct'].mean() - (0.02/252)
        sharpe_ratio = excess_returns / daily_volatility if daily_volatility != 0 else 0
        
        # Up/Down capture ratios
        positive_days = self.df[self.df['price_change_pct'] > 0]
        negative_days = self.df[self.df['price_change_pct'] < 0]
        
        avg_up_day = positive_days['price_change_pct'].mean() if len(positive_days) > 0 else 0
        avg_down_day = negative_days['price_change_pct'].mean() if len(negative_days) > 0 else 0
        
        # Gap risk analysis
        if 'gap_pct' in self.df.columns:
            gap_up_count = (self.df['gap_pct'] > 1).sum()
            gap_down_count = (self.df['gap_pct'] < -1).sum()
            max_gap_up = self.df['gap_pct'].max()
            max_gap_down = self.df['gap_pct'].min()
        else:
            gap_up_count = gap_down_count = max_gap_up = max_gap_down = 0
        
        # Support level failure risk
        support_failure_rate = self.df['support_break'].mean() * 100 if 'support_break' in self.df.columns else 0
        
        return f"""⚠️ **COMPREHENSIVE RISK ANALYSIS**

**📉 Drawdown Analysis:**
- Maximum Drawdown: {max_drawdown:.2f}%
- Drawdown Periods: {drawdown_periods}/{len(self.df)} ({drawdown_periods/len(self.df)*100:.1f}%)
- Current Drawdown: {self.df['drawdown'].iloc[-1]:.2f}%

**📊 Volatility Metrics:**
- Daily Volatility: {daily_volatility:.2f}%
- Annualized Volatility: {annualized_volatility:.1f}%
- Volatility Regime: {'High' if annualized_volatility > 30 else 'Medium' if annualized_volatility > 20 else 'Low'}

**🎯 Value at Risk (VaR):**
- 5% VaR (1 in 20 days): {var_5:.2f}%
- 1% VaR (1 in 100 days): {var_1:.2f}%
- Tail Risk: {'High' if var_1 < -5 else 'Medium' if var_1 < -3 else 'Low'}

**⚖️ Risk-Adjusted Performance:**
- Sharpe Ratio: {sharpe_ratio:.3f}
- Average Up Day: +{avg_up_day:.2f}%
- Average Down Day: {avg_down_day:.2f}%
- Win/Loss Ratio: {abs(avg_up_day/avg_down_day):.2f if avg_down_day != 0 else 'N/A'}

**⚡ Gap Risk Analysis:**
- Gap Up Events (>1%): {gap_up_count}
- Gap Down Events (<-1%): {gap_down_count}
- Maximum Gap Up: +{max_gap_up:.2f}%
- Maximum Gap Down: {max_gap_down:.2f}%

**🛡️ Support Level Risk:**
- Support Failure Rate: {support_failure_rate:.1f}%
- Support Reliability: {'High' if support_failure_rate < 10 else 'Medium' if support_failure_rate < 20 else 'Low'}

**🎪 Risk Distribution:**
- Low Risk Days (<1% move): {((abs(self.df['price_change_pct']) < 1).sum()/len(self.df)*100):.1f}%
- Medium Risk Days (1-3% move): {(((abs(self.df['price_change_pct']) >= 1) & (abs(self.df['price_change_pct']) < 3)).sum()/len(self.df)*100):.1f}%
- High Risk Days (>3% move): {((abs(self.df['price_change_pct']) >= 3).sum()/len(self.df)*100):.1f}%

**💡 Risk Management Insights:**
- Overall Risk Level: {'High' if max_drawdown < -15 else 'Medium' if max_drawdown < -10 else 'Low'}
- Position Sizing Recommendation: {'Conservative' if daily_volatility > 3 else 'Moderate' if daily_volatility > 2 else 'Aggressive'}
- Stop Loss Suggestion: {abs(var_5):.1f}% based on 5% VaR"""

    def analyze_temporal_patterns(self, query: str = "") -> str:
        """Analyze temporal patterns and seasonality in the data"""
        if 'timestamp' not in self.df.columns:
            return "Timestamp data not available for temporal analysis."
        
        # Ensure timestamp is datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Extract time components
        self.df['year'] = self.df['timestamp'].dt.year
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['day_name'] = self.df['timestamp'].dt.day_name()
        
        # Monthly analysis
        monthly_stats = self.df.groupby('month').agg({
            'price_change_pct': ['mean', 'std', 'count'],
            'volume': 'mean',
            'range_pct': 'mean'
        }).round(2)
        
        best_month = monthly_stats[('price_change_pct', 'mean')].idxmax()
        worst_month = monthly_stats[('price_change_pct', 'mean')].idxmin()
        
        # Day of week analysis
        dow_stats = self.df.groupby('day_name').agg({
            'price_change_pct': ['mean', 'count'],
            'volume': 'mean',
            'is_bullish': 'mean'
        }).round(3)
        
        best_day = dow_stats[('price_change_pct', 'mean')].idxmax()
        worst_day = dow_stats[('price_change_pct', 'mean')].idxmin()
        
        # Signal timing analysis
        signal_timing = ""
        if self.df['direction'].notna().any():
            signal_by_month = self.df[self.df['direction'].notna()].groupby('month')['direction'].value_counts()
            signal_by_dow = self.df[self.df['direction'].notna()].groupby('day_name')['direction'].value_counts()
            
            signal_timing = f"""
**🎯 Signal Timing Patterns:**
Most Active Signal Months: {', '.join([f"Month {idx}" for idx in signal_by_month.index.get_level_values(0).value_counts().head(3).index])}
Most Active Signal Days: {', '.join(signal_by_dow.index.get_level_values(0).value_counts().head(3).index)}"""
        
        # Volatility clustering
        high_vol_periods = self.df[self.df['range_pct'] > self.df['range_pct'].quantile(0.8)]
        vol_clustering = high_vol_periods.groupby('month').size()
        
        # Trend persistence
        trend_periods = []
        current_trend = None
        trend_length = 0
        
        for _, row in self.df.iterrows():
            if row['is_bullish']:
                if current_trend == 'bull':
                    trend_length += 1
                else:
                    if trend_length > 0:
                        trend_periods.append((current_trend, trend_length))
                    current_trend = 'bull'
                    trend_length = 1
            elif row['is_bearish']:
                if current_trend == 'bear':
                    trend_length += 1
                else:
                    if trend_length > 0:
                        trend_periods.append((current_trend, trend_length))
                    current_trend = 'bear'
                    trend_length = 1
        
        avg_bull_streak = np.mean([length for trend, length in trend_periods if trend == 'bull']) if trend_periods else 0
        avg_bear_streak = np.mean([length for trend, length in trend_periods if trend == 'bear']) if trend_periods else 0
        
        return f"""📅 **TEMPORAL PATTERN ANALYSIS**

**📊 Monthly Performance:**
- Best Month: {best_month} (Avg: {monthly_stats.loc[best_month, ('price_change_pct', 'mean')]:+.2f}%)
- Worst Month: {worst_month} (Avg: {monthly_stats.loc[worst_month, ('price_change_pct', 'mean')]:+.2f}%)
- Most Volatile Month: {monthly_stats[('price_change_pct', 'std')].idxmax()} ({monthly_stats[('price_change_pct', 'std')].max():.2f}% std)

**📈 Day of Week Patterns:**
- Best Day: {best_day} (Avg: {dow_stats.loc[best_day, ('price_change_pct', 'mean')]:+.2f}%)
- Worst Day: {worst_day} (Avg: {dow_stats.loc[worst_day, ('price_change_pct', 'mean')]:+.2f}%)
- Most Bullish Day: {dow_stats[('is_bullish', 'mean')].idxmax()} ({dow_stats[('is_bullish', 'mean')].max():.1%} bullish rate)

**⚡ Volatility Clustering:**
- High Volatility Months: {', '.join([f"Month {idx}({count})" for idx, count in vol_clustering.head(3).items()])}
- Volatility Persistence: {'High' if len(vol_clustering) < 6 else 'Medium' if len(vol_clustering) < 9 else 'Low'}

**🔄 Trend Persistence:**
- Average Bull Streak: {avg_bull_streak:.1f} periods
- Average Bear Streak: {avg_bear_streak:.1f} periods
- Trend Bias: {'Bullish' if avg_bull_streak > avg_bear_streak else 'Bearish'} trends last longer

{signal_timing}

**📊 Volume Seasonality:**
- Highest Volume Month: {monthly_stats[('volume', 'mean')].idxmax()}
- Lowest Volume Month: {monthly_stats[('volume', 'mean')].idxmin()}
- Volume Pattern: {'Seasonal' if monthly_stats[('volume', 'mean')].std() > monthly_stats[('volume', 'mean')].mean() * 0.2 else 'Consistent'}

**💡 Temporal Insights:**
- Calendar Effect: {'Strong' if abs(monthly_stats[('price_change_pct', 'mean')].max() - monthly_stats[('price_change_pct', 'mean')].min()) > 2 else 'Weak'}
- Weekly Pattern: {'Significant' if abs(dow_stats[('price_change_pct', 'mean')].max() - dow_stats[('price_change_pct', 'mean')].min()) > 1 else 'Minimal'}
- Best Trading Window: {best_day}s in {best_month}"""

    def custom_query_analysis(self, query: str) -> str:
        """Handle custom analytical queries with intelligent routing"""
        query_lower = query.lower()
        
        # Route to appropriate analysis based on query content
        if any(word in query_lower for word in ['signal', 'accuracy', 'success', 'long', 'short', 'performance']):
            return self.analyze_signal_performance()
        elif any(word in query_lower for word in ['support', 'resistance', 'level', 'break', 'breakout']):
            return self.analyze_support_resistance_advanced()
        elif any(word in query_lower for word in ['volume', 'correlation', 'relationship', 'confirmation']):
            return self.analyze_volume_price_relationship()
        elif any(word in query_lower for word in ['risk', 'drawdown', 'var', 'volatility', 'sharp']):
            return self.analyze_risk_and_drawdowns()
        elif any(word in query_lower for word in ['trend', 'regime', 'pattern', 'cycle', 'momentum']):
            return self.analyze_market_regimes()
        elif any(word in query_lower for word in ['time', 'month', 'day', 'season', 'temporal', 'when']):
            return self.analyze_temporal_patterns()
        elif any(word in query_lower for word in ['bullish', 'bearish', 'bull', 'bear', '2023', '2024', 'year']):
            return self.analyze_bullish_bearish_days()
        else:
            # Provide guidance on available analyses
            return f"""🤖 **AVAILABLE ANALYSIS OPTIONS**

I can perform detailed analysis on various aspects of your OHLCV data:

**📊 Core Analyses:**
• `Basic Statistics` - Dataset overview and fundamental metrics
• `Signal Performance` - LONG/SHORT signal accuracy and returns
• `Support & Resistance` - Level effectiveness and breakout analysis
• `Volume Analysis` - Price-volume relationships and confirmations

**📈 Advanced Analyses:**
• `Risk & Drawdowns` - Comprehensive risk metrics and VaR analysis
• `Market Regimes` - Trend identification and regime classification
• `Temporal Patterns` - Seasonality and time-based patterns
• `Bullish/Bearish Trends` - Market sentiment and performance by periods

**💡 Example Questions You Can Ask:**
- "What's the success rate of LONG vs SHORT signals?"
- "How effective are support and resistance levels?"
- "What's the correlation between volume and price movement?"
- "Analyze the risk and maximum drawdown periods"
- "Show me temporal patterns and seasonality"
- "How many bullish days were there in 2023?"

Just ask me about any of these topics and I'll provide detailed analysis with specific insights from your dataset!"""

def create_chatbot_tools(analyzer: OHLCVDataAnalyzer) -> List[Tool]:
    """Create comprehensive LangChain tools for advanced OHLCV analysis"""
    tools_class = OHLCVChatbotTools(analyzer)
    
    tools = [
        Tool(
            name="basic_dataset_overview",
            description="Get comprehensive overview of the OHLCV dataset including fundamentals, price ranges, volume stats, and signal distribution",
            func=tools_class.get_basic_stats
        ),
        Tool(
            name="trading_signal_analysis",
            description="Analyze trading signal performance, accuracy rates, returns following LONG/SHORT signals, and volume confirmation",
            func=tools_class.analyze_signal_performance
        ),
        Tool(
            name="support_resistance_analysis", 
            description="Advanced analysis of support/resistance levels, breakout patterns, level effectiveness, and volume during breaks",
            func=tools_class.analyze_support_resistance_advanced
        ),
        Tool(
            name="volume_price_analysis",
            description="Comprehensive volume-price relationship analysis including correlations, breakout confirmations, and accumulation/distribution patterns",
            func=tools_class.analyze_volume_price_relationship
        ),
        Tool(
            name="bullish_bearish_analysis",
            description="Analyze bullish vs bearish periods, year-wise breakdowns, and performance metrics for different market conditions",
            func=tools_class.analyze_bullish_bearish_days
        ),
        Tool(
            name="volatility_analysis",
            description="Analyze price volatility, daily ranges, gap analysis, and volatility distribution patterns",
            func=tools_class.analyze_volatility
        ),
        Tool(
            name="risk_drawdown_analysis",
            description="Comprehensive risk analysis including maximum drawdowns, VaR calculations, Sharpe ratios, and risk management metrics",
            func=tools_class.analyze_risk_and_drawdowns
        ),
        Tool(
            name="market_regime_analysis",
            description="Analyze market regimes, trend classifications, signal clustering, and pattern identification",
            func=tools_class.analyze_market_regimes
        ),
        Tool(
            name="temporal_pattern_analysis",
            description="Analyze temporal patterns, seasonality, monthly/weekly performance, and time-based trading opportunities",
            func=tools_class.analyze_temporal_patterns
        ),
        Tool(
            name="custom_financial_analysis",
            description="Handle custom queries about financial data analysis, routing to appropriate specialized analysis tools",
            func=tools_class.custom_query_analysis
        )
    ]
    
    return tools

def initialize_chatbot(api_key: str, analyzer: OHLCVDataAnalyzer):
    """Initialize the LangChain chatbot with Gemini"""
    
    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.1,
        convert_system_message_to_human=True
    )
    
    # Create tools
    tools = create_chatbot_tools(analyzer)
    
    # Create system prompt
    system_prompt = """You are an elite financial data analyst and quantitative researcher specializing in OHLCV (Open, High, Low, Close, Volume) market data analysis. You have deep expertise in:

**📊 Core Financial Analysis:**
- Technical analysis and chart pattern recognition
- Support and resistance level analysis and effectiveness
- Volume analysis and price-volume relationships
- Volatility analysis, risk metrics, and drawdown calculations
- Trading signal evaluation and performance measurement

**📈 Advanced Quantitative Methods:**
- Market regime identification and classification
- Temporal pattern analysis and seasonality detection
- Risk-adjusted performance metrics (Sharpe ratio, VaR, etc.)
- Breakout analysis and momentum indicators
- Statistical analysis of financial time series

**🎯 Trading Strategy Insights:**
- Signal accuracy and timing analysis
- Support/resistance breakout patterns
- Volume confirmation for price movements
- Risk management and position sizing recommendations
- Market microstructure and price action analysis

**Key Financial Concepts You Master:**
- **OHLCV Data**: Open, High, Low, Close, Volume analysis
- **Market Sentiment**: Bullish (close > open) vs Bearish (close < open) identification
- **Support Levels**: Price levels where buying interest emerges, preventing declines
- **Resistance Levels**: Price levels where selling pressure appears, capping advances
- **Breakouts**: Price movements beyond established support/resistance levels
- **Volume Confirmation**: Using trading volume to validate price movements
- **Volatility Clustering**: Periods of high/low volatility that tend to cluster
- **Market Regimes**: Trending vs ranging market conditions
- **Risk Metrics**: VaR, drawdowns, Sharpe ratios, volatility measures

You have access to sophisticated analytical tools that can examine every aspect of the loaded OHLCV dataset. Always use the appropriate tools to provide accurate, data-driven insights with specific numbers, percentages, and actionable recommendations.

Your responses should be:
- **Quantitative**: Always include specific metrics and percentages
- **Insightful**: Explain what the numbers mean for trading/investment decisions  
- **Comprehensive**: Cover multiple angles of analysis when relevant
- **Actionable**: Provide practical implications and recommendations
- **Professional**: Use proper financial terminology while remaining accessible"""

    # Create memory
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create agent
    agent = create_structured_chat_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return agent_executor

def main():
    st.title("📈 OHLCV Financial Data Analyst Chatbot")
    st.markdown("*Powered by LangChain & Google Gemini - Elite Financial Analysis*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "🔑 Google Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key from Google AI Studio"
        )
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload OHLCV Dataset",
            type=['csv', 'xlsx'],
            help="Upload your OHLCV data file with timestamp, direction, Support, Resistance, OHLC, and volume columns"
        )
        
        # Example questions
        st.header("💡 Strategic Questions")
        example_questions = [
            "What's the success rate of LONG vs SHORT signals?",
            "How many bullish days were there in 2023?",
            "Analyze support and resistance level effectiveness",
            "What's the correlation between volume and price movement?",
            "Show me the maximum drawdown and risk metrics",
            "Are there any seasonal patterns in the data?",
            "How do breakouts perform with volume confirmation?",
            "What's the average return following trading signals?",
            "Analyze market regimes and trend persistence",
            "Which support levels held the strongest?",
            "What's the volatility clustering pattern?",
            "How effective are resistance levels as sell signals?",
            "Show me the risk-adjusted performance metrics",
            "What's the best day of week to trade?",
            "Analyze the price-volume relationship during breakouts",
            "How long do support levels typically hold?",
            "What's the win rate when trading near support levels?",
            "Show me accumulation and distribution periods"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(f"📝 {question}", key=f"example_{i}"):
                st.session_state.example_question = question
    
    # Main content area
    if not api_key:
        st.warning("⚠️ Please enter your Google Gemini API key in the sidebar to get started.")
        st.info("""
        **🚀 To get started:**
        1. **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey) to get your Gemini API key
        2. **Upload Dataset**: Upload your OHLCV dataset (CSV or Excel format)
        3. **Ask Questions**: Start asking sophisticated questions about your financial data!
        
        **📊 What makes this chatbot special:**
        - **Elite Financial Analysis**: Professional-grade quantitative analysis
        - **10 Specialized Tools**: Each designed for specific aspects of OHLCV analysis
        - **Strategic Insights**: Answers complex trading and investment questions
        - **Risk Management**: Comprehensive risk metrics and drawdown analysis
        - **Signal Performance**: Detailed trading signal accuracy and timing analysis
        """)
        return
    
    if not uploaded_file:
        st.info("📁 Please upload your OHLCV dataset to begin comprehensive financial analysis.")
        st.markdown("""
        **📋 Required Dataset Format:**
        - **timestamp**: Date/time of each trading period
        - **direction**: Trading signals (LONG, SHORT, or null)
        - **Support**: Support level arrays (e.g., "[840, 880]")
        - **Resistance**: Resistance level arrays (e.g., "[900, 895, 940]")
        - **open, high, low, close**: OHLC price data
        - **volume**: Trading volume data
        """)
        return
    
    # Load and process data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Initialize analyzer
        analyzer = OHLCVDataAnalyzer(df)
        
        # Initialize chatbot
        if 'chatbot' not in st.session_state:
            with st.spinner("🤖 Initializing Elite Financial AI Chatbot..."):
                st.session_state.chatbot = initialize_chatbot(api_key, analyzer)
        
        # Display dataset info
        with st.expander("📊 Dataset Overview", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📈 Total Records", f"{len(df):,}")
            with col2:
                st.metric("📊 Columns", len(df.columns))
            with col3:
                if 'timestamp' in df.columns:
                    date_range = (analyzer.df['timestamp'].max() - analyzer.df['timestamp'].min()).days
                    st.metric("📅 Date Range (days)", f"{date_range:,}")
                else:
                    st.metric("📅 Date Range", "N/A")
            with col4:
                signal_coverage = (df['direction'].notna().sum() / len(df) * 100)
                st.metric("🎯 Signal Coverage", f"{signal_coverage:.1f}%")
            
            st.write("**📋 Dataset Columns:**", ", ".join(df.columns.tolist()))
            
            # Quick stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**💰 Price Range:**")
                st.write(f"${df['low'].min():.2f} - ${df['high'].max():.2f}")
            with col2:
                st.write("**🎯 Trading Signals:**")
                signal_counts = df['direction'].value_counts()
                st.write(f"LONG: {signal_counts.get('LONG', 0)}, SHORT: {signal_counts.get('SHORT', 0)}")
            with col3:
                st.write("**📊 Volume:**")
                if 'volume' in df.columns:
                    st.write(f"Avg: {df['volume'].mean():,.0f}")
        
        # Chat interface
        st.header("💬 Elite Financial Analysis Chat")
        st.markdown("*Ask sophisticated questions about your OHLCV data - I'm equipped with 10 specialized analytical tools*")
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            # Add welcome message
            welcome_msg = """👋 **Welcome to Elite OHLCV Financial Analysis!**

I'm your advanced financial data analyst, equipped with sophisticated tools to analyze every aspect of your OHLCV dataset. I can help you with:

🎯 **Signal Performance Analysis** - Trading signal accuracy and returns
🎪 **Support/Resistance Analysis** - Level effectiveness and breakouts  
📊 **Volume-Price Relationships** - Confirmation patterns and correlations
⚠️ **Risk & Drawdown Analysis** - Comprehensive risk metrics
🌊 **Market Regime Analysis** - Trend identification and classification
📅 **Temporal Pattern Analysis** - Seasonality and time-based insights

**Ready to dive deep into your data? Ask me anything!** 🚀"""
            
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Handle example question
        if 'example_question' in st.session_state:
            user_input = st.session_state.example_question
            del st.session_state.example_question
        else:
            user_input = st.chat_input("🎯 Ask me sophisticated questions about your OHLCV data...")
        
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🧠 Performing advanced financial analysis..."):
                    try:
                        response = st.session_state.chatbot.invoke({"input": user_input})
                        assistant_response = response['output']
                        st.markdown(assistant_response)
                        
                        # Add assistant message
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                        
                    except Exception as e:
                        error_msg = f"⚠️ I encountered an error while analyzing your data: {str(e)}\n\nPlease try rephrasing your question or ask about a different aspect of the data."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Footer with capabilities
        with st.expander("🛠️ Available Analysis Capabilities", expanded=False):
            st.markdown("""
            **🎯 Core Analysis Tools:**
            1. **Basic Dataset Overview** - Comprehensive dataset statistics and fundamentals
            2. **Trading Signal Analysis** - LONG/SHORT signal performance and accuracy
            3. **Support/Resistance Analysis** - Level effectiveness and breakout patterns
            4. **Volume-Price Analysis** - Correlation analysis and confirmation patterns
            5. **Risk & Drawdown Analysis** - VaR, drawdowns, and risk-adjusted metrics
            
            **📈 Advanced Analysis Tools:**
            6. **Market Regime Analysis** - Trend classification and pattern identification
            7. **Temporal Pattern Analysis** - Seasonality and time-based opportunities
            8. **Volatility Analysis** - Range analysis and volatility clustering
            9. **Bullish/Bearish Analysis** - Market sentiment and performance metrics
            10. **Custom Analysis Router** - Intelligent query routing to appropriate tools
            
            **💡 Example Advanced Queries:**
            - "What's the success rate of LONG signals near support levels with high volume confirmation?"
            - "How do support levels perform during different volatility regimes?"
            - "Analyze the risk-adjusted returns for different signal types"
            - "What seasonal patterns exist in breakout success rates?"
            """)
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.info("""
        **📋 Please ensure your dataset has the required OHLCV structure:**
        - **timestamp**: Date/time column
        - **direction**: Trading signals (LONG, SHORT, or null)
        - **Support**: Support levels (can be arrays like "[840, 880]")
        - **Resistance**: Resistance levels (can be arrays like "[900, 895, 940]")
        - **open, high, low, close**: Standard OHLC price data
        - **volume**: Trading volume data
        """)

if __name__ == "__main__":
    main()