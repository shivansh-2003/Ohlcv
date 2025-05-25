import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from data_analyzer import OHLCVDataAnalyzer
from chatbot_tools import initialize_chatbot

# Set page config
st.set_page_config(
    page_title="OHLCV Financial Data Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .example-query {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load and cache the TSLA data"""
    try:
        df = pd.read_csv('data/TSLA_data.csv')
        return df
    except FileNotFoundError:
        st.error("❌ TSLA_data.csv not found in data/ directory")
        return None

def create_price_chart(df):
    """Create an interactive price chart with volume"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('TSLA Price Action', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="TSLA"
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['red' if close < open else 'green' for close, open in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="TSLA OHLCV Data Overview",
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=False
    )
    
    return fig

def create_signal_chart(df):
    """Create a chart showing trading signals"""
    fig = go.Figure()
    
    # Price line
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        )
    )
    
    # LONG signals
    long_signals = df[df['direction'] == 'LONG']
    if len(long_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=long_signals['timestamp'],
                y=long_signals['close'],
                mode='markers',
                name='LONG Signals',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green'
                )
            )
        )
    
    # SHORT signals
    short_signals = df[df['direction'] == 'SHORT']
    if len(short_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=short_signals['timestamp'],
                y=short_signals['close'],
                mode='markers',
                name='SHORT Signals',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red'
                )
            )
        )
    
    fig.update_layout(
        title="Trading Signals on TSLA",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Advanced OHLCV Financial Analyst</h1>', unsafe_allow_html=True)
    st.markdown("### Sophisticated AI-Powered Trading Analysis with Tesla (TSLA) Data")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key to enable the AI chatbot"
        )
        
        st.markdown("---")
        
        # Dataset info
        st.header("📊 Dataset Info")
        st.info("""
        **Tesla (TSLA) OHLCV Data**
        - Time Period: Aug 2022 - Jun 2023
        - Trading Signals: LONG/SHORT
        - Support/Resistance Levels
        - Volume Data
        - 665+ Trading Periods
        """)
        
        st.markdown("---")
        
        # Example queries
        st.header("💡 Example Queries")
        example_queries = [
            "What's the success rate of LONG signals near support levels?",
            "How do support levels perform during high volatility?",
            "Analyze volume confirmation patterns",
            "What seasonal patterns exist in the data?",
            "Calculate risk-adjusted returns by market regime",
            "Show me the correlation between volume and volatility"
        ]
        
        for query in example_queries:
            st.markdown(f'<div class="example-query">"{query}"</div>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Initialize analyzer
    analyzer = OHLCVDataAnalyzer(df)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🤖 AI Analyst", "📈 Charts", "📋 Data"])
    
    with tab1:
        st.header("📊 Dataset Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Records",
                f"{len(df):,}",
                help="Total number of trading periods"
            )
        
        with col2:
            signal_count = df['direction'].notna().sum()
            st.metric(
                "Trading Signals",
                f"{signal_count:,}",
                help="Number of LONG/SHORT signals"
            )
        
        with col3:
            price_range = df['high'].max() - df['low'].min()
            st.metric(
                "Price Range",
                f"${price_range:.2f}",
                help="Total price range in dataset"
            )
        
        with col4:
            if 'volume' in df.columns:
                total_volume = df['volume'].sum()
                st.metric(
                    "Total Volume",
                    f"{total_volume:,.0f}",
                    help="Total volume traded"
                )
        
        # Quick stats
        st.subheader("📈 Quick Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Signal distribution
            signal_dist = df['direction'].value_counts()
            fig_signals = px.pie(
                values=signal_dist.values,
                names=signal_dist.index,
                title="Trading Signal Distribution",
                color_discrete_map={'LONG': 'green', 'SHORT': 'red'}
            )
            st.plotly_chart(fig_signals, use_container_width=True)
        
        with col2:
            # Price movement distribution
            bullish_days = (df['close'] > df['open']).sum()
            bearish_days = (df['close'] < df['open']).sum()
            doji_days = len(df) - bullish_days - bearish_days
            
            fig_movement = px.pie(
                values=[bullish_days, bearish_days, doji_days],
                names=['Bullish', 'Bearish', 'Doji'],
                title="Daily Price Movement Distribution",
                color_discrete_map={'Bullish': 'green', 'Bearish': 'red', 'Doji': 'gray'}
            )
            st.plotly_chart(fig_movement, use_container_width=True)
    
    with tab2:
        st.header("🤖 AI Financial Analyst")
        
        if not api_key:
            st.warning("⚠️ Please enter your Google Gemini API key in the sidebar to use the AI analyst.")
            st.info("""
            **Get your API key:**
            1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Create a new API key
            3. Copy and paste it in the sidebar
            """)
        else:
            # Initialize chatbot
            if 'chatbot' not in st.session_state:
                with st.spinner("🔄 Initializing AI Analyst..."):
                    try:
                        st.session_state.chatbot = initialize_chatbot(api_key, analyzer)
                        st.success("✅ AI Analyst ready!")
                    except Exception as e:
                        st.error(f"❌ Failed to initialize chatbot: {str(e)}")
                        st.info("Please check your API key and try again.")
                        return
            
            # Chat interface
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Chat history
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask me anything about the TSLA data..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Analyzing..."):
                        try:
                            response = st.session_state.chatbot.invoke({"input": prompt})
                            ai_response = response['output']
                            st.markdown(ai_response)
                            st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        except Exception as e:
                            error_msg = f"❌ Error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Quick action buttons
            st.subheader("🚀 Quick Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Dataset Overview", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": "Give me a comprehensive overview of the dataset"})
                    st.rerun()
            
            with col2:
                if st.button("🎯 Signal Performance", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": "Analyze the trading signal performance"})
                    st.rerun()
            
            with col3:
                if st.button("📈 Volume Analysis", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": "Analyze the volume-price relationship"})
                    st.rerun()
    
    with tab3:
        st.header("📈 Interactive Charts")
        
        # Price chart
        st.subheader("💰 Price Action & Volume")
        price_chart = create_price_chart(df)
        st.plotly_chart(price_chart, use_container_width=True)
        
        # Signal chart
        st.subheader("🎯 Trading Signals")
        signal_chart = create_signal_chart(df)
        st.plotly_chart(signal_chart, use_container_width=True)
        
        # Additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily returns distribution
            daily_returns = ((df['close'] - df['open']) / df['open'] * 100)
            fig_returns = px.histogram(
                daily_returns,
                title="Daily Returns Distribution",
                labels={'value': 'Daily Return (%)', 'count': 'Frequency'},
                nbins=50
            )
            st.plotly_chart(fig_returns, use_container_width=True)
        
        with col2:
            # Volume over time
            fig_volume = px.line(
                df,
                x='timestamp',
                y='volume',
                title="Volume Over Time"
            )
            st.plotly_chart(fig_volume, use_container_width=True)
    
    with tab4:
        st.header("📋 Raw Data")
        
        # Data filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_signals_only = st.checkbox("Show only periods with signals")
        
        with col2:
            signal_type = st.selectbox("Signal Type", ["All", "LONG", "SHORT"])
        
        with col3:
            # Fix the date range input to handle timestamp properly
            if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                min_date = df['timestamp'].min().date()
                max_date = df['timestamp'].max().date()
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            else:
                st.info("Date filtering not available - timestamp column not properly formatted")
                date_range = None
        
        # Filter data
        filtered_df = df.copy()
        
        if show_signals_only:
            filtered_df = filtered_df[filtered_df['direction'].notna()]
        
        if signal_type != "All":
            filtered_df = filtered_df[filtered_df['direction'] == signal_type]
        
        # Apply date filtering only if date_range is available and properly formatted
        if date_range and len(date_range) == 2 and 'timestamp' in filtered_df.columns:
            try:
                # Ensure timestamp column is datetime
                if not pd.api.types.is_datetime64_any_dtype(filtered_df['timestamp']):
                    filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
                
                # Convert date_range to datetime for comparison
                start_date = pd.Timestamp(date_range[0])
                end_date = pd.Timestamp(date_range[1])
                
                filtered_df = filtered_df[
                    (filtered_df['timestamp'] >= start_date) &
                    (filtered_df['timestamp'] <= end_date)
                ]
            except Exception as e:
                st.warning(f"Date filtering failed: {str(e)}")
        
        # Display data
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="tsla_filtered_data.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        🚀 Advanced OHLCV Financial Analyst | Powered by AI & Tesla Data
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 