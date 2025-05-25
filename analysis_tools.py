import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import Counter
from data_analyzer import OHLCVDataAnalyzer


class OHLCVAnalysisTools:
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
        """Analyze trading signal performance with sophisticated metrics"""
        signals_df = self.df[self.df['direction'].notna()].copy()
        
        if len(signals_df) == 0:
            return "❌ No trading signals found in the dataset."
        
        # Calculate signal accuracy based on next period performance
        signals_df['next_close'] = signals_df['close'].shift(-1)
        signals_df['signal_success'] = (
            ((signals_df['direction'] == 'LONG') & (signals_df['next_close'] > signals_df['close'])) |
            ((signals_df['direction'] == 'SHORT') & (signals_df['next_close'] < signals_df['close']))
        )
        
        # Overall performance metrics
        total_signals = len(signals_df)
        successful_signals = signals_df['signal_success'].sum()
        success_rate = (successful_signals / total_signals) * 100
        
        # Performance by signal type
        long_signals = signals_df[signals_df['direction'] == 'LONG']
        short_signals = signals_df[signals_df['direction'] == 'SHORT']
        
        long_success_rate = (long_signals['signal_success'].sum() / len(long_signals)) * 100 if len(long_signals) > 0 else 0
        short_success_rate = (short_signals['signal_success'].sum() / len(short_signals)) * 100 if len(short_signals) > 0 else 0
        
        # Volume analysis for signals
        high_volume_signals = signals_df[signals_df['high_volume'] == True] if 'high_volume' in signals_df.columns else pd.DataFrame()
        hv_success_rate = (high_volume_signals['signal_success'].sum() / len(high_volume_signals)) * 100 if len(high_volume_signals) > 0 else 0
        
        # Support/Resistance proximity analysis
        near_support = signals_df[signals_df['support_distance'] <= 5] if 'support_distance' in signals_df.columns else pd.DataFrame()
        near_resistance = signals_df[signals_df['resistance_distance'] <= 5] if 'resistance_distance' in signals_df.columns else pd.DataFrame()
        
        support_success_rate = (near_support['signal_success'].sum() / len(near_support)) * 100 if len(near_support) > 0 else 0
        resistance_success_rate = (near_resistance['signal_success'].sum() / len(near_resistance)) * 100 if len(near_resistance) > 0 else 0
        
        # Risk-reward analysis
        signals_df['return_pct'] = ((signals_df['next_close'] - signals_df['close']) / signals_df['close']) * 100
        signals_df['directional_return'] = np.where(
            signals_df['direction'] == 'LONG', 
            signals_df['return_pct'], 
            -signals_df['return_pct']
        )
        
        avg_return = signals_df['directional_return'].mean()
        win_rate = (signals_df['directional_return'] > 0).sum() / len(signals_df) * 100
        avg_win = signals_df[signals_df['directional_return'] > 0]['directional_return'].mean()
        avg_loss = signals_df[signals_df['directional_return'] < 0]['directional_return'].mean()
        
        return f"""🎯 **ADVANCED SIGNAL PERFORMANCE ANALYSIS**

**📊 Overall Performance:**
- Total Signals Analyzed: {total_signals}
- Success Rate: {success_rate:.1f}% ({successful_signals}/{total_signals})
- Average Return per Signal: {avg_return:.2f}%
- Win Rate: {win_rate:.1f}%

**📈 Signal Type Performance:**
- LONG Signals: {len(long_signals)} | Success Rate: {long_success_rate:.1f}%
- SHORT Signals: {len(short_signals)} | Success Rate: {short_success_rate:.1f}%

**🔊 Volume-Confirmed Signals:**
- High Volume Signals: {len(high_volume_signals)}
- High Volume Success Rate: {hv_success_rate:.1f}%

**🎪 Support/Resistance Proximity:**
- Near Support Signals: {len(near_support)} | Success Rate: {support_success_rate:.1f}%
- Near Resistance Signals: {len(near_resistance)} | Success Rate: {resistance_success_rate:.1f}%

**💰 Risk-Reward Metrics:**
- Average Winning Trade: {avg_win:.2f}%
- Average Losing Trade: {avg_loss:.2f}%
- Risk-Reward Ratio: {abs(avg_win/avg_loss):.2f}:1 (if avg_loss != 0)

**🔍 Key Insights:**
- Best performing signal type: {'LONG' if long_success_rate > short_success_rate else 'SHORT'}
- Volume confirmation {'improves' if hv_success_rate > success_rate else 'does not improve'} success rate
- Support level signals {'outperform' if support_success_rate > success_rate else 'underperform'} average"""

    def analyze_support_resistance_advanced(self, query: str = "") -> str:
        """Advanced support and resistance level analysis"""
        # Filter data with S/R levels
        sr_data = self.df[
            (self.df['support_levels'].apply(len) > 0) | 
            (self.df['resistance_levels'].apply(len) > 0)
        ].copy()
        
        if len(sr_data) == 0:
            return "❌ No support/resistance data available for analysis."
        
        # Support level analysis
        support_data = sr_data[sr_data['support_levels'].apply(len) > 0]
        resistance_data = sr_data[sr_data['resistance_levels'].apply(len) > 0]
        
        # Breakout analysis
        support_breaks = self.df[self.df['support_break'] == True] if 'support_break' in self.df.columns else pd.DataFrame()
        resistance_breaks = self.df[self.df['resistance_break'] == True] if 'resistance_break' in self.df.columns else pd.DataFrame()
        
        # Price action near levels
        near_support = self.df[self.df['support_distance'] <= 2] if 'support_distance' in self.df.columns else pd.DataFrame()
        near_resistance = self.df[self.df['resistance_distance'] <= 2] if 'resistance_distance' in self.df.columns else pd.DataFrame()
        
        # Bounce analysis
        support_bounces = 0
        resistance_bounces = 0
        
        if len(near_support) > 0:
            support_bounces = (near_support['close'] > near_support['open']).sum()
        
        if len(near_resistance) > 0:
            resistance_bounces = (near_resistance['close'] < near_resistance['open']).sum()
        
        # Volume at S/R levels
        sr_volume_analysis = ""
        if 'volume' in self.df.columns:
            support_avg_volume = near_support['volume'].mean() if len(near_support) > 0 else 0
            resistance_avg_volume = near_resistance['volume'].mean() if len(near_resistance) > 0 else 0
            overall_avg_volume = self.df['volume'].mean()
            
            sr_volume_analysis = f"""
**📊 Volume at S/R Levels:**
- Average Volume at Support: {support_avg_volume:,.0f} ({(support_avg_volume/overall_avg_volume-1)*100:+.1f}% vs average)
- Average Volume at Resistance: {resistance_avg_volume:,.0f} ({(resistance_avg_volume/overall_avg_volume-1)*100:+.1f}% vs average)"""
        
        # Signal performance near S/R levels
        signals_near_support = self.df[
            (self.df['direction'].notna()) & 
            (self.df['support_distance'] <= 3)
        ] if 'support_distance' in self.df.columns else pd.DataFrame()
        
        signals_near_resistance = self.df[
            (self.df['direction'].notna()) & 
            (self.df['resistance_distance'] <= 3)
        ] if 'resistance_distance' in self.df.columns else pd.DataFrame()
        
        return f"""🎪 **ADVANCED SUPPORT & RESISTANCE ANALYSIS**

**📊 Level Coverage:**
- Periods with Support Levels: {len(support_data)}/{len(self.df)} ({len(support_data)/len(self.df)*100:.1f}%)
- Periods with Resistance Levels: {len(resistance_data)}/{len(self.df)} ({len(resistance_data)/len(self.df)*100:.1f}%)
- Average S/R Range: ${self.df['sr_range'].mean():.2f}

**💥 Breakout Analysis:**
- Support Breakdowns: {len(support_breaks)} events
- Resistance Breakouts: {len(resistance_breaks)} events
- Total Breakout Rate: {(len(support_breaks) + len(resistance_breaks))/len(sr_data)*100:.1f}%

**🎯 Price Action Near Levels:**
- Periods Near Support (≤$2): {len(near_support)}
- Support Bounces: {support_bounces}/{len(near_support)} ({support_bounces/len(near_support)*100:.1f}% if len(near_support) > 0 else 0)
- Periods Near Resistance (≤$2): {len(near_resistance)}
- Resistance Rejections: {resistance_bounces}/{len(near_resistance)} ({resistance_bounces/len(near_resistance)*100:.1f}% if len(near_resistance) > 0 else 0)

{sr_volume_analysis}

**🎯 Signal Performance Near S/R:**
- Signals Near Support: {len(signals_near_support)}
- Signals Near Resistance: {len(signals_near_resistance)}

**🔍 Key Insights:**
- Support levels show {support_bounces/len(near_support)*100:.1f}% bounce rate
- Resistance levels show {resistance_bounces/len(near_resistance)*100:.1f}% rejection rate
- {'High' if (len(support_breaks) + len(resistance_breaks))/len(sr_data) > 0.1 else 'Low'} breakout frequency detected"""

    def analyze_volume_price_relationship(self, query: str = "") -> str:
        """Analyze the relationship between volume and price movements"""
        if 'volume' not in self.df.columns:
            return "❌ Volume data not available for analysis."
        
        # Volume-Price correlation
        volume_price_corr = self.df['volume'].corr(self.df['price_change_pct'])
        volume_range_corr = self.df['volume'].corr(self.df['range_pct'])
        
        # High volume analysis
        high_vol_threshold = self.df['volume'].quantile(0.8)
        high_volume_days = self.df[self.df['volume'] > high_vol_threshold]
        
        # Volume patterns
        bullish_high_vol = high_volume_days[high_volume_days['is_bullish'] == True]
        bearish_high_vol = high_volume_days[high_volume_days['is_bearish'] == True]
        
        # Volume during different market conditions
        trending_up = self.df[self.df['price_change_pct'] > 2]
        trending_down = self.df[self.df['price_change_pct'] < -2]
        ranging = self.df[abs(self.df['price_change_pct']) <= 2]
        
        avg_vol_up = trending_up['volume'].mean() if len(trending_up) > 0 else 0
        avg_vol_down = trending_down['volume'].mean() if len(trending_down) > 0 else 0
        avg_vol_range = ranging['volume'].mean() if len(ranging) > 0 else 0
        
        # Volume confirmation analysis
        volume_confirmed_moves = self.df[
            ((self.df['price_change_pct'] > 1) & (self.df['volume'] > self.df['volume_sma'])) |
            ((self.df['price_change_pct'] < -1) & (self.df['volume'] > self.df['volume_sma']))
        ]
        
        # Breakout volume analysis
        breakouts = self.df[
            (self.df['support_break'] == True) | (self.df['resistance_break'] == True)
        ] if 'support_break' in self.df.columns else pd.DataFrame()
        
        breakout_volume_analysis = ""
        if len(breakouts) > 0:
            avg_breakout_volume = breakouts['volume'].mean()
            overall_avg = self.df['volume'].mean()
            breakout_volume_analysis = f"""
**💥 Breakout Volume Analysis:**
- Breakout Events: {len(breakouts)}
- Average Breakout Volume: {avg_breakout_volume:,.0f}
- Volume Premium on Breakouts: {(avg_breakout_volume/overall_avg-1)*100:+.1f}%"""
        
        return f"""📊 **VOLUME-PRICE RELATIONSHIP ANALYSIS**

**🔗 Correlation Analysis:**
- Volume-Price Change Correlation: {volume_price_corr:.3f}
- Volume-Range Correlation: {volume_range_corr:.3f}
- Relationship Strength: {'Strong' if abs(volume_price_corr) > 0.3 else 'Moderate' if abs(volume_price_corr) > 0.1 else 'Weak'}

**🔊 High Volume Analysis (Top 20%):**
- High Volume Days: {len(high_volume_days)}
- Bullish High Volume: {len(bullish_high_vol)} ({len(bullish_high_vol)/len(high_volume_days)*100:.1f}%)
- Bearish High Volume: {len(bearish_high_vol)} ({len(bearish_high_vol)/len(high_volume_days)*100:.1f}%)

**📈 Volume by Market Condition:**
- Trending Up Days (>2%): {len(trending_up)} | Avg Volume: {avg_vol_up:,.0f}
- Trending Down Days (<-2%): {len(trending_down)} | Avg Volume: {avg_vol_down:,.0f}
- Ranging Days (±2%): {len(ranging)} | Avg Volume: {avg_vol_range:,.0f}

**✅ Volume Confirmation:**
- Volume-Confirmed Moves: {len(volume_confirmed_moves)}
- Confirmation Rate: {len(volume_confirmed_moves)/len(self.df)*100:.1f}%

{breakout_volume_analysis}

**🔍 Key Insights:**
- {'Positive' if volume_price_corr > 0 else 'Negative'} volume-price correlation detected
- High volume days are {len(bullish_high_vol)/len(high_volume_days)*100:.1f}% bullish
- {'Up' if avg_vol_up > avg_vol_down else 'Down'} trends show higher average volume
- Volume confirmation occurs in {len(volume_confirmed_moves)/len(self.df)*100:.1f}% of significant moves"""

    def analyze_bullish_bearish_days(self, query: str = "") -> str:
        """Analyze bullish vs bearish day patterns and characteristics"""
        bullish_days = self.df[self.df['is_bullish'] == True]
        bearish_days = self.df[self.df['is_bearish'] == True]
        doji_days = self.df[self.df['is_bullish'] == self.df['is_bearish']]  # Neither bullish nor bearish
        
        total_days = len(self.df)
        bullish_pct = len(bullish_days) / total_days * 100
        bearish_pct = len(bearish_days) / total_days * 100
        doji_pct = len(doji_days) / total_days * 100
        
        # Average characteristics
        avg_bullish_change = bullish_days['price_change_pct'].mean() if len(bullish_days) > 0 else 0
        avg_bearish_change = bearish_days['price_change_pct'].mean() if len(bearish_days) > 0 else 0
        
        # Volume analysis
        volume_analysis = ""
        if 'volume' in self.df.columns:
            avg_bullish_volume = bullish_days['volume'].mean() if len(bullish_days) > 0 else 0
            avg_bearish_volume = bearish_days['volume'].mean() if len(bearish_days) > 0 else 0
            overall_avg_volume = self.df['volume'].mean()
            
            volume_analysis = f"""
**📊 Volume Characteristics:**
- Average Bullish Day Volume: {avg_bullish_volume:,.0f} ({(avg_bullish_volume/overall_avg_volume-1)*100:+.1f}% vs average)
- Average Bearish Day Volume: {avg_bearish_volume:,.0f} ({(avg_bearish_volume/overall_avg_volume-1)*100:+.1f}% vs average)"""
        
        # Range analysis
        avg_bullish_range = bullish_days['range_pct'].mean() if len(bullish_days) > 0 else 0
        avg_bearish_range = bearish_days['range_pct'].mean() if len(bearish_days) > 0 else 0
        
        return f"""📊 **BULLISH vs BEARISH DAYS ANALYSIS**

**📈 Day Distribution:**
- Bullish Days: {len(bullish_days)} ({bullish_pct:.1f}%)
- Bearish Days: {len(bearish_days)} ({bearish_pct:.1f}%)
- Doji/Neutral Days: {len(doji_days)} ({doji_pct:.1f}%)

**💰 Price Movement Characteristics:**
- Average Bullish Day Change: {avg_bullish_change:.2f}%
- Average Bearish Day Change: {avg_bearish_change:.2f}%
- Average Bullish Range: {avg_bullish_range:.2f}%
- Average Bearish Range: {avg_bearish_range:.2f}%

{volume_analysis}

**🔍 Market Sentiment:**
- Overall Bias: {'Bullish' if bullish_pct > bearish_pct else 'Bearish' if bearish_pct > bullish_pct else 'Neutral'}
- Sentiment Strength: {abs(bullish_pct - bearish_pct):.1f}% difference
- Volatility: {'Higher on bearish days' if avg_bearish_range > avg_bullish_range else 'Higher on bullish days'}"""

    def analyze_volatility(self, query: str = "") -> str:
        """Comprehensive volatility analysis"""
        # Calculate various volatility measures
        self.df['volatility'] = self.df['range_pct']
        
        # Rolling volatility
        self.df['volatility_20'] = self.df['volatility'].rolling(window=20).std()
        self.df['volatility_5'] = self.df['volatility'].rolling(window=5).std()
        
        # Volatility regimes
        vol_median = self.df['volatility'].median()
        vol_75th = self.df['volatility'].quantile(0.75)
        vol_25th = self.df['volatility'].quantile(0.25)
        
        low_vol = self.df[self.df['volatility'] <= vol_25th]
        medium_vol = self.df[(self.df['volatility'] > vol_25th) & (self.df['volatility'] <= vol_75th)]
        high_vol = self.df[self.df['volatility'] > vol_75th]
        
        # Performance in different volatility regimes
        low_vol_performance = low_vol['price_change_pct'].mean() if len(low_vol) > 0 else 0
        medium_vol_performance = medium_vol['price_change_pct'].mean() if len(medium_vol) > 0 else 0
        high_vol_performance = high_vol['price_change_pct'].mean() if len(high_vol) > 0 else 0
        
        # Volatility clustering analysis
        high_vol_days = self.df[self.df['volatility'] > vol_75th]
        volatility_clusters = 0
        if len(high_vol_days) > 1:
            # Count consecutive high volatility days
            high_vol_indices = high_vol_days.index.tolist()
            clusters = []
            current_cluster = [high_vol_indices[0]]
            
            for i in range(1, len(high_vol_indices)):
                if high_vol_indices[i] - high_vol_indices[i-1] == 1:
                    current_cluster.append(high_vol_indices[i])
                else:
                    if len(current_cluster) > 1:
                        clusters.append(current_cluster)
                    current_cluster = [high_vol_indices[i]]
            
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
            
            volatility_clusters = len(clusters)
        
        return f"""📊 **COMPREHENSIVE VOLATILITY ANALYSIS**

**📈 Volatility Distribution:**
- Average Daily Volatility: {self.df['volatility'].mean():.2f}%
- Median Volatility: {vol_median:.2f}%
- Volatility Range: {self.df['volatility'].min():.2f}% - {self.df['volatility'].max():.2f}%

**🎯 Volatility Regimes:**
- Low Volatility Days (≤25th percentile): {len(low_vol)} ({len(low_vol)/len(self.df)*100:.1f}%)
- Medium Volatility Days: {len(medium_vol)} ({len(medium_vol)/len(self.df)*100:.1f}%)
- High Volatility Days (≥75th percentile): {len(high_vol)} ({len(high_vol)/len(self.df)*100:.1f}%)

**💰 Performance by Volatility Regime:**
- Low Volatility Performance: {low_vol_performance:.2f}%
- Medium Volatility Performance: {medium_vol_performance:.2f}%
- High Volatility Performance: {high_vol_performance:.2f}%

**🔗 Volatility Clustering:**
- High Volatility Clusters Detected: {volatility_clusters}
- Clustering Tendency: {'Strong' if volatility_clusters > len(high_vol)/4 else 'Moderate' if volatility_clusters > 0 else 'Weak'}

**🔍 Key Insights:**
- Best performing regime: {'Low' if low_vol_performance > max(medium_vol_performance, high_vol_performance) else 'Medium' if medium_vol_performance > high_vol_performance else 'High'} volatility
- Volatility shows {'strong' if volatility_clusters > len(high_vol)/3 else 'moderate'} clustering behavior
- Current volatility level: {'High' if self.df['volatility'].iloc[-1] > vol_75th else 'Low' if self.df['volatility'].iloc[-1] <= vol_25th else 'Medium'}"""

    def analyze_market_regimes(self, query: str = "") -> str:
        """Analyze different market regimes and their characteristics"""
        # Define market regimes based on price action and volatility
        self.df['price_ma_20'] = self.df['close'].rolling(window=20).mean()
        self.df['trend_direction'] = np.where(
            self.df['close'] > self.df['price_ma_20'], 'Uptrend',
            np.where(self.df['close'] < self.df['price_ma_20'], 'Downtrend', 'Sideways')
        )
        
        # Volatility regimes
        vol_median = self.df['range_pct'].median()
        self.df['vol_regime'] = np.where(
            self.df['range_pct'] > vol_median, 'High Vol', 'Low Vol'
        )
        
        # Combined regimes
        self.df['market_regime'] = self.df['trend_direction'] + ' + ' + self.df['vol_regime']
        
        # Analyze each regime
        regime_analysis = {}
        for regime in self.df['market_regime'].unique():
            if pd.notna(regime):
                regime_data = self.df[self.df['market_regime'] == regime]
                regime_analysis[regime] = {
                    'count': len(regime_data),
                    'avg_return': regime_data['price_change_pct'].mean(),
                    'avg_volume': regime_data['volume'].mean() if 'volume' in regime_data.columns else 0,
                    'success_rate': 0
                }
                
                # Signal success rate in this regime
                regime_signals = regime_data[regime_data['direction'].notna()]
                if len(regime_signals) > 0:
                    regime_signals['next_close'] = regime_signals['close'].shift(-1)
                    regime_signals['signal_success'] = (
                        ((regime_signals['direction'] == 'LONG') & (regime_signals['next_close'] > regime_signals['close'])) |
                        ((regime_signals['direction'] == 'SHORT') & (regime_signals['next_close'] < regime_signals['close']))
                    )
                    regime_analysis[regime]['success_rate'] = regime_signals['signal_success'].mean() * 100
        
        # Format results
        regime_summary = ""
        for regime, stats in regime_analysis.items():
            regime_summary += f"""
**{regime}:**
- Periods: {stats['count']} ({stats['count']/len(self.df)*100:.1f}%)
- Avg Return: {stats['avg_return']:.2f}%
- Avg Volume: {stats['avg_volume']:,.0f}
- Signal Success Rate: {stats['success_rate']:.1f}%"""
        
        # Best and worst regimes
        best_regime = max(regime_analysis.keys(), key=lambda x: regime_analysis[x]['avg_return'])
        worst_regime = min(regime_analysis.keys(), key=lambda x: regime_analysis[x]['avg_return'])
        
        return f"""🎯 **MARKET REGIME ANALYSIS**

**📊 Regime Classification:**
- Trend: Based on 20-period moving average
- Volatility: Based on median daily range

{regime_summary}

**🏆 Performance Summary:**
- Best Performing Regime: {best_regime} ({regime_analysis[best_regime]['avg_return']:.2f}% avg return)
- Worst Performing Regime: {worst_regime} ({regime_analysis[worst_regime]['avg_return']:.2f}% avg return)

**🔍 Key Insights:**
- Most common regime: {max(regime_analysis.keys(), key=lambda x: regime_analysis[x]['count'])}
- Highest signal success: {max(regime_analysis.keys(), key=lambda x: regime_analysis[x]['success_rate'])} ({max(regime_analysis.values(), key=lambda x: x['success_rate'])['success_rate']:.1f}%)
- Current regime: {self.df['market_regime'].iloc[-1] if pd.notna(self.df['market_regime'].iloc[-1]) else 'Unknown'}"""

    def analyze_risk_and_drawdowns(self, query: str = "") -> str:
        """Comprehensive risk and drawdown analysis"""
        # Calculate cumulative returns
        self.df['daily_return'] = self.df['price_change_pct'] / 100
        self.df['cumulative_return'] = (1 + self.df['daily_return']).cumprod()
        
        # Calculate rolling maximum (peak)
        self.df['rolling_max'] = self.df['cumulative_return'].expanding().max()
        
        # Calculate drawdown
        self.df['drawdown'] = (self.df['cumulative_return'] / self.df['rolling_max']) - 1
        
        # Risk metrics
        total_return = (self.df['cumulative_return'].iloc[-1] - 1) * 100
        volatility = self.df['daily_return'].std() * np.sqrt(252) * 100  # Annualized
        sharpe_ratio = (self.df['daily_return'].mean() * 252) / (self.df['daily_return'].std() * np.sqrt(252)) if self.df['daily_return'].std() > 0 else 0
        
        # Drawdown analysis
        max_drawdown = self.df['drawdown'].min() * 100
        avg_drawdown = self.df[self.df['drawdown'] < 0]['drawdown'].mean() * 100 if len(self.df[self.df['drawdown'] < 0]) > 0 else 0
        
        # Drawdown duration analysis
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(self.df['drawdown']):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                drawdown_periods.append(i - start_idx)
        
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Value at Risk (VaR) - 5% and 1%
        var_5 = np.percentile(self.df['daily_return'], 5) * 100
        var_1 = np.percentile(self.df['daily_return'], 1) * 100
        
        # Conditional Value at Risk (CVaR)
        cvar_5 = self.df[self.df['daily_return'] <= np.percentile(self.df['daily_return'], 5)]['daily_return'].mean() * 100
        cvar_1 = self.df[self.df['daily_return'] <= np.percentile(self.df['daily_return'], 1)]['daily_return'].mean() * 100
        
        # Upside/Downside analysis
        positive_days = self.df[self.df['daily_return'] > 0]
        negative_days = self.df[self.df['daily_return'] < 0]
        
        upside_capture = positive_days['daily_return'].mean() * 100 if len(positive_days) > 0 else 0
        downside_capture = negative_days['daily_return'].mean() * 100 if len(negative_days) > 0 else 0
        
        # Risk-adjusted metrics by signal
        signal_risk_analysis = ""
        signals_df = self.df[self.df['direction'].notna()].copy()
        if len(signals_df) > 0:
            signals_df['next_return'] = signals_df['daily_return'].shift(-1)
            signals_df['directional_return'] = np.where(
                signals_df['direction'] == 'LONG',
                signals_df['next_return'],
                -signals_df['next_return']
            )
            
            signal_sharpe = signals_df['directional_return'].mean() / signals_df['directional_return'].std() if signals_df['directional_return'].std() > 0 else 0
            signal_var = np.percentile(signals_df['directional_return'], 5) * 100
            
            signal_risk_analysis = f"""
**🎯 Signal-Based Risk Metrics:**
- Signal Sharpe Ratio: {signal_sharpe:.3f}
- Signal VaR (5%): {signal_var:.2f}%
- Signal Win Rate: {(signals_df['directional_return'] > 0).mean() * 100:.1f}%"""
        
        return f"""⚠️ **COMPREHENSIVE RISK & DRAWDOWN ANALYSIS**

**📊 Return Metrics:**
- Total Return: {total_return:.2f}%
- Annualized Volatility: {volatility:.2f}%
- Sharpe Ratio: {sharpe_ratio:.3f}

**📉 Drawdown Analysis:**
- Maximum Drawdown: {max_drawdown:.2f}%
- Average Drawdown: {avg_drawdown:.2f}%
- Number of Drawdown Periods: {len(drawdown_periods)}
- Average Drawdown Duration: {avg_drawdown_duration:.1f} periods
- Maximum Drawdown Duration: {max_drawdown_duration} periods

**⚠️ Value at Risk (VaR):**
- 1-Day VaR (5%): {var_5:.2f}%
- 1-Day VaR (1%): {var_1:.2f}%
- Conditional VaR (5%): {cvar_5:.2f}%
- Conditional VaR (1%): {cvar_1:.2f}%

**📈 Upside/Downside Analysis:**
- Positive Days: {len(positive_days)} ({len(positive_days)/len(self.df)*100:.1f}%)
- Negative Days: {len(negative_days)} ({len(negative_days)/len(self.df)*100:.1f}%)
- Average Upside: {upside_capture:.2f}%
- Average Downside: {downside_capture:.2f}%
- Upside/Downside Ratio: {abs(upside_capture/downside_capture):.2f}

{signal_risk_analysis}

**🔍 Risk Assessment:**
- Risk Level: {'High' if volatility > 30 else 'Medium' if volatility > 15 else 'Low'}
- Drawdown Severity: {'Severe' if max_drawdown < -20 else 'Moderate' if max_drawdown < -10 else 'Mild'}
- Recovery Ability: {'Poor' if avg_drawdown_duration > 10 else 'Good'}"""

    def analyze_temporal_patterns(self, query: str = "") -> str:
        """Analyze temporal patterns in the data"""
        if 'timestamp' not in self.df.columns:
            return "❌ Timestamp data not available for temporal analysis."
        
        # Extract time components
        self.df['year'] = self.df['timestamp'].dt.year
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['day_of_month'] = self.df['timestamp'].dt.day
        
        # Monthly analysis
        monthly_stats = self.df.groupby('month').agg({
            'price_change_pct': ['mean', 'std', 'count'],
            'volume': 'mean' if 'volume' in self.df.columns else lambda x: 0
        }).round(2)
        
        # Day of week analysis
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_stats = self.df.groupby('day_of_week').agg({
            'price_change_pct': ['mean', 'std', 'count'],
            'volume': 'mean' if 'volume' in self.df.columns else lambda x: 0
        }).round(2)
        
        # Best and worst performing periods
        best_month = monthly_stats['price_change_pct']['mean'].idxmax()
        worst_month = monthly_stats['price_change_pct']['mean'].idxmin()
        best_dow = dow_stats['price_change_pct']['mean'].idxmax()
        worst_dow = dow_stats['price_change_pct']['mean'].idxmin()
        
        # Seasonal signal analysis
        seasonal_signal_analysis = ""
        if self.df['direction'].notna().any():
            signal_seasonal = self.df[self.df['direction'].notna()].groupby('month').agg({
                'direction': 'count'
            }).rename(columns={'direction': 'signal_count'})
            
            peak_signal_month = signal_seasonal['signal_count'].idxmax()
            seasonal_signal_analysis = f"""
**🎯 Seasonal Signal Patterns:**
- Peak Signal Month: {peak_signal_month} ({signal_seasonal.loc[peak_signal_month, 'signal_count']} signals)
- Signal Distribution: Varies by month"""
        
        # Volatility patterns
        monthly_vol = self.df.groupby('month')['range_pct'].mean()
        dow_vol = self.df.groupby('day_of_week')['range_pct'].mean()
        
        most_volatile_month = monthly_vol.idxmax()
        most_volatile_dow = dow_vol.idxmax()
        
        # Format monthly performance
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_performance = ""
        for month in range(1, 13):
            if month in monthly_stats.index:
                perf = monthly_stats.loc[month, ('price_change_pct', 'mean')]
                count = monthly_stats.loc[month, ('price_change_pct', 'count')]
                monthly_performance += f"- {month_names[month-1]}: {perf:.2f}% ({count} days)\n"
        
        # Format day of week performance
        dow_performance = ""
        for dow in range(7):
            if dow in dow_stats.index:
                perf = dow_stats.loc[dow, ('price_change_pct', 'mean')]
                count = dow_stats.loc[dow, ('price_change_pct', 'count')]
                dow_performance += f"- {dow_names[dow]}: {perf:.2f}% ({count} days)\n"
        
        return f"""📅 **TEMPORAL PATTERNS ANALYSIS**

**📊 Monthly Performance:**
{monthly_performance}

**📅 Day of Week Performance:**
{dow_performance}

**🏆 Best/Worst Periods:**
- Best Month: {month_names[best_month-1]} ({monthly_stats.loc[best_month, ('price_change_pct', 'mean')]:.2f}%)
- Worst Month: {month_names[worst_month-1]} ({monthly_stats.loc[worst_month, ('price_change_pct', 'mean')]:.2f}%)
- Best Day: {dow_names[best_dow]} ({dow_stats.loc[best_dow, ('price_change_pct', 'mean')]:.2f}%)
- Worst Day: {dow_names[worst_dow]} ({dow_stats.loc[worst_dow, ('price_change_pct', 'mean')]:.2f}%)

**📊 Volatility Patterns:**
- Most Volatile Month: {month_names[most_volatile_month-1]} ({monthly_vol[most_volatile_month]:.2f}% avg range)
- Most Volatile Day: {dow_names[most_volatile_dow]} ({dow_vol[most_volatile_dow]:.2f}% avg range)

{seasonal_signal_analysis}

**🔍 Key Insights:**
- Seasonal bias: {'Positive' if monthly_stats['price_change_pct']['mean'].mean() > 0 else 'Negative'}
- Weekly pattern: {'Exists' if dow_stats['price_change_pct']['mean'].std() > 0.5 else 'Minimal'}
- Volatility clustering: {'Monthly' if monthly_vol.std() > dow_vol.std() else 'Weekly'}"""

    def custom_query_analysis(self, query: str) -> str:
        """Handle custom analytical queries with intelligent parsing"""
        query_lower = query.lower()
        
        # Query routing based on keywords
        if any(word in query_lower for word in ['support', 'resistance', 'level']):
            return self.analyze_support_resistance_advanced(query)
        elif any(word in query_lower for word in ['volume', 'confirmation']):
            return self.analyze_volume_price_relationship(query)
        elif any(word in query_lower for word in ['signal', 'success', 'performance']):
            return self.analyze_signal_performance(query)
        elif any(word in query_lower for word in ['volatility', 'regime']):
            if 'regime' in query_lower:
                return self.analyze_market_regimes(query)
            else:
                return self.analyze_volatility(query)
        elif any(word in query_lower for word in ['risk', 'drawdown', 'var']):
            return self.analyze_risk_and_drawdowns(query)
        elif any(word in query_lower for word in ['seasonal', 'temporal', 'month', 'day']):
            return self.analyze_temporal_patterns(query)
        elif any(word in query_lower for word in ['bullish', 'bearish']):
            return self.analyze_bullish_bearish_days(query)
        else:
            # Default comprehensive analysis
            return f"""🤖 **CUSTOM ANALYSIS REQUEST**

I understand you're asking: "{query}"

Let me provide a comprehensive analysis that might address your question:

{self.get_basic_stats()}

For more specific analysis, try asking about:
- Signal performance and success rates
- Support/resistance level analysis  
- Volume-price relationships
- Market regimes and volatility
- Risk metrics and drawdowns
- Temporal/seasonal patterns

Would you like me to focus on any particular aspect?""" 