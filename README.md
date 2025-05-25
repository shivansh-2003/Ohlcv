# 🚀 Advanced OHLCV Financial Analyst

A sophisticated AI-powered financial analysis chatbot that provides deep insights into OHLCV (Open, High, Low, Close, Volume) trading data. Built with a modular architecture using LangChain, Google Gemini AI, and Streamlit.

## ✨ Features

### 🤖 Intelligent AI Analyst
- **Advanced Query Processing**: Handles sophisticated financial queries with intelligent routing
- **Multi-Tool Analysis**: Uses specialized tools for different types of analysis
- **Conversational Interface**: Natural language interaction with memory and context
- **Real-time Insights**: Provides actionable trading intelligence

### 📊 Comprehensive Analysis Capabilities

#### 🎯 Signal Performance Analysis
- Success rate calculations for LONG/SHORT signals
- Risk-reward metrics and win rates
- Volume confirmation analysis
- Support/resistance proximity effects

#### 🎪 Support & Resistance Analysis
- Breakout and breakdown detection
- Bounce rate calculations
- Volume analysis at key levels
- Signal performance near S/R levels

#### 📈 Volume-Price Relationship
- Correlation analysis between volume and price movements
- Volume confirmation patterns
- Breakout volume analysis
- Market condition volume behavior

#### ⚡ Market Regime Analysis
- Trend and volatility regime classification
- Performance metrics by regime
- Signal success rates across different conditions

#### ⚠️ Risk & Drawdown Analysis
- Comprehensive risk metrics (VaR, CVaR, Sharpe ratio)
- Drawdown analysis and duration
- Upside/downside capture ratios
- Signal-based risk assessment

#### 📅 Temporal Pattern Analysis
- Seasonal performance patterns
- Monthly and weekly analysis
- Volatility clustering detection
- Time-based signal distribution

### 🎨 Modern UI Features
- **Interactive Charts**: Candlestick charts, volume analysis, signal visualization
- **Real-time Metrics**: Key performance indicators and statistics
- **Data Filtering**: Advanced filtering and export capabilities
- **Responsive Design**: Modern, mobile-friendly interface

## 🏗️ Modular Architecture

The application is built with a clean, modular architecture:

```
├── data_analyzer.py      # Core OHLCV data processing and technical indicators
├── analysis_tools.py     # Comprehensive analysis methods and calculations
├── chatbot_tools.py      # LangChain tool creation and chatbot initialization
├── streamlit_app.py      # Main Streamlit application and UI
├── requirements.txt      # Python dependencies
└── data/
    └── TSLA_data.csv    # Tesla OHLCV dataset
```

### 📦 Module Descriptions

#### `data_analyzer.py`
- **OHLCVDataAnalyzer**: Core class for data preparation and basic technical indicators
- Handles timestamp conversion, support/resistance parsing
- Calculates price changes, moving averages, volume metrics

#### `analysis_tools.py`
- **OHLCVAnalysisTools**: Advanced analysis methods for sophisticated queries
- 10+ specialized analysis functions
- Intelligent query routing and custom analysis handling

#### `chatbot_tools.py`
- **Tool Creation**: LangChain tool definitions and descriptions
- **Chatbot Initialization**: Agent setup with memory and conversation handling
- **System Prompts**: Sophisticated prompts for financial analysis

#### `streamlit_app.py`
- **Main Application**: Complete Streamlit interface
- **Interactive Charts**: Plotly-based visualizations
- **Chat Interface**: Real-time AI conversation
- **Data Management**: Filtering, export, and display features

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ohlcv-financial-analyst

# Install dependencies
pip install -r requirements.txt
```

### 2. Get Google Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key for use in the application

### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

### 4. Start Analyzing

1. Enter your Google Gemini API key in the sidebar
2. Explore the dataset overview
3. Ask sophisticated questions in the AI Analyst tab
4. View interactive charts and raw data

## 💡 Example Queries

The chatbot can handle sophisticated financial analysis queries:

### 🎯 Signal Analysis
- "What's the success rate of LONG signals near support levels with high volume confirmation?"
- "How do SHORT signals perform during high volatility periods?"
- "Analyze the risk-adjusted returns for different signal types"

### 🎪 Support/Resistance Analysis
- "How do support levels perform during different volatility regimes?"
- "What's the breakout success rate when volume is above average?"
- "Analyze bounce rates at resistance levels"

### 📊 Market Conditions
- "Calculate the correlation between volume and price volatility during trending vs ranging markets"
- "What are the characteristics of high-performing market regimes?"
- "How does volatility clustering affect signal performance?"

### 📅 Temporal Patterns
- "What seasonal patterns exist in breakout success rates?"
- "Analyze monthly performance variations"
- "How do signals perform on different days of the week?"

## 📊 Dataset

The application uses Tesla (TSLA) OHLCV data with:
- **Time Period**: August 2022 - June 2023
- **665+ Trading Periods**
- **LONG/SHORT Trading Signals**
- **Support/Resistance Levels**
- **Volume Data**

### Data Structure
```csv
timestamp,direction,Support,Resistance,open,high,low,close,volume
2022-08-25,SHORT,"[840, 880]","[900, 895, 940]",302.542,302.807,291.556,295.897,6685.7261
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit with custom CSS styling
- **AI/ML**: Google Gemini Pro via LangChain
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly (Candlestick charts, interactive plots)
- **Agent Framework**: LangChain with structured chat agents
- **Memory**: Conversation buffer with context retention

## 🔧 Configuration

### Environment Variables (Optional)
Create a `.env` file for default API key:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Customization
- **Add New Analysis Tools**: Extend `OHLCVAnalysisTools` class
- **Modify UI**: Update `streamlit_app.py` styling and layout
- **Change Data Source**: Replace TSLA data with your own OHLCV dataset
- **Adjust AI Behavior**: Modify system prompts in `chatbot_tools.py`

## 📈 Key Capabilities

### Advanced Analytics
- **Signal Performance**: Success rates, risk-reward ratios, volume confirmation
- **Market Regimes**: Trend + volatility classification with performance metrics
- **Risk Management**: VaR, drawdowns, Sharpe ratios, upside/downside analysis
- **Temporal Analysis**: Seasonal patterns, volatility clustering, time-based insights

### AI-Powered Insights
- **Natural Language Queries**: Ask complex questions in plain English
- **Intelligent Routing**: Automatically selects appropriate analysis tools
- **Contextual Memory**: Maintains conversation context for follow-up questions
- **Actionable Recommendations**: Provides practical trading insights

### Interactive Visualizations
- **Candlestick Charts**: Professional trading chart with volume
- **Signal Overlays**: Visual representation of LONG/SHORT signals
- **Distribution Analysis**: Returns, volume, and performance distributions
- **Time Series**: Trend analysis and pattern recognition

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Tesla (TSLA)** for the sample financial data
- **Google Gemini** for advanced AI capabilities
- **LangChain** for the agent framework
- **Streamlit** for the web application framework
- **Plotly** for interactive financial charts

---

**Built with ❤️ for financial analysis and algorithmic trading** 