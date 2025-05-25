from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import List
from data_analyzer import OHLCVDataAnalyzer
from analysis_tools import OHLCVAnalysisTools


def create_chatbot_tools(analyzer: OHLCVDataAnalyzer) -> List[Tool]:
    """Create comprehensive LangChain tools for OHLCV analysis"""
    tools_instance = OHLCVAnalysisTools(analyzer)
    
    tools = [
        Tool(
            name="get_basic_stats",
            description="Get comprehensive overview and basic statistics of the OHLCV dataset including price ranges, volume, signals distribution, and support/resistance coverage",
            func=tools_instance.get_basic_stats
        ),
        Tool(
            name="analyze_signal_performance",
            description="Analyze trading signal performance with success rates, risk-reward metrics, volume confirmation analysis, and support/resistance proximity effects. Perfect for queries about signal accuracy and effectiveness.",
            func=tools_instance.analyze_signal_performance
        ),
        Tool(
            name="analyze_support_resistance_advanced",
            description="Advanced analysis of support and resistance levels including breakout analysis, bounce rates, volume at key levels, and signal performance near S/R levels. Use for queries about support/resistance effectiveness.",
            func=tools_instance.analyze_support_resistance_advanced
        ),
        Tool(
            name="analyze_volume_price_relationship",
            description="Comprehensive analysis of volume-price relationships including correlation analysis, volume confirmation patterns, breakout volume analysis, and volume behavior in different market conditions.",
            func=tools_instance.analyze_volume_price_relationship
        ),
        Tool(
            name="analyze_bullish_bearish_days",
            description="Analyze characteristics of bullish vs bearish trading days including distribution, average movements, volume patterns, and market sentiment analysis.",
            func=tools_instance.analyze_bullish_bearish_days
        ),
        Tool(
            name="analyze_volatility",
            description="Comprehensive volatility analysis including volatility regimes, clustering analysis, performance in different volatility environments, and current volatility assessment.",
            func=tools_instance.analyze_volatility
        ),
        Tool(
            name="analyze_market_regimes",
            description="Analyze different market regimes combining trend direction and volatility levels. Shows performance and signal success rates across different market conditions.",
            func=tools_instance.analyze_market_regimes
        ),
        Tool(
            name="analyze_risk_and_drawdowns",
            description="Comprehensive risk analysis including drawdown metrics, Value at Risk (VaR), Sharpe ratios, upside/downside analysis, and signal-based risk metrics.",
            func=tools_instance.analyze_risk_and_drawdowns
        ),
        Tool(
            name="analyze_temporal_patterns",
            description="Analyze seasonal and temporal patterns including monthly/weekly performance, volatility patterns, and seasonal signal distribution. Perfect for seasonal analysis queries.",
            func=tools_instance.analyze_temporal_patterns
        ),
        Tool(
            name="custom_query_analysis",
            description="Handle custom analytical queries with intelligent routing to appropriate analysis methods. Use this for complex or specific questions that don't fit other tools.",
            func=tools_instance.custom_query_analysis
        )
    ]
    
    return tools


def initialize_chatbot(api_key: str, analyzer: OHLCVDataAnalyzer):
    """Initialize the advanced OHLCV chatbot with comprehensive analysis capabilities"""
    
    # Initialize the LLM with updated model name
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Updated model name
        google_api_key=api_key,
        temperature=0.1
        # Removed deprecated convert_system_message_to_human parameter
    )
    
    # Create tools
    tools = create_chatbot_tools(analyzer)
    
    # Create a simple ReAct prompt template
    template = """You are an advanced financial data analyst specializing in OHLCV (Open, High, Low, Close, Volume) data analysis. You have access to sophisticated analytical tools that can provide deep insights into trading patterns, market behavior, and signal performance.

Your Capabilities:
- Comprehensive signal performance analysis with success rates and risk metrics
- Advanced support/resistance level analysis with breakout and bounce detection
- Volume-price relationship analysis with confirmation patterns
- Market regime analysis combining trend and volatility
- Risk and drawdown analysis with VaR calculations
- Temporal pattern analysis for seasonal insights
- Volatility analysis with regime classification
- Custom query handling with intelligent routing

Guidelines:
1. ALWAYS use tools to provide data-driven insights rather than making assumptions
2. Be specific and actionable in your responses
3. Highlight key insights and practical implications
4. Use multiple tools when needed to provide comprehensive analysis
5. Explain complex concepts in accessible terms
6. Focus on practical trading insights and risk management

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    
    # Create agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return agent_executor 