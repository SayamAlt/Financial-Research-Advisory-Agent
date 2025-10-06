from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List, TypedDict, Literal
import yfinance as yf
import numpy as np
import warnings, requests, os, json, datetime
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from langchain_openai import ChatOpenAI
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

load_dotenv()

if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
if "NEWS_API_KEY" in st.secrets:
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
else:
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
if "SEC_EDGAR_API_KEY" in st.secrets: 
    SEC_EDGAR_API_KEY = st.secrets["SEC_EDGAR_API_KEY"]
else:
    SEC_EDGAR_API_KEY = os.getenv("SEC_EDGAR_API_KEY")

llm = ChatOpenAI(model="gpt-4", temperature=0.5)

if "MONGO_URI" in st.secrets:
    uri = st.secrets["MONGO_URI"]
else:
    uri = os.getenv("MONGO_URI")
    
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['stockadvicedb']

market_data_col = db['market_data']
news_col = db['financial_news']
filings_col = db['sec_filings']
feature_data_col = db['feature_data']
signal_col = db['signals']
compliance_col = db['compliance']
advice_col = db['financial_advice']
human_review_col = db['human_review']
audit_col = db['audit_logs']

class MarketData(BaseModel):
    symbol: str
    last_close: float
    volume: int
    open: float
    low: float
    high: float

class NewsState(BaseModel):
    headlines: List[str]
    summaries: List[str]

class FeatureData(BaseModel):
    symbol: str
    moving_average_50: float
    moving_average_200: float
    rsi: float
    macd: float
    volatility: float
    risk_score: float

class FinanceData(TypedDict):
    symbol: str
    market_data: MarketData
    news: NewsState
    feature_data: FeatureData
    filings: List[str]
    strategy: str
    signal: str
    confidence: float
    accuracy: float
    compliance: str
    compliance_reason: str
    risk: str
    advice: str
    audit: str

def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    else:
        return obj
def get_market_data(state: FinanceData):
    symbol = state.get("symbol", "AAPL")
    data = yf.download(symbol, period="1d", interval="1d")
    last_day = data.tail(1).iloc[0]
    record = {
        "timestamp": str(datetime.datetime.now()),
        "action": "get_market_data",
        "symbol": symbol,
        "details": json.loads(data.tail(1).to_json())
    }
    market_data_col.insert_one(record)
    return to_serializable({
        "market_data": MarketData(
            symbol=symbol,
            last_close=float(last_day["Close"]),
            volume=int(last_day["Volume"]),
            open=float(last_day["Open"]),
            low=float(last_day["Low"]),
            high=float(last_day["High"]),
        )
    })

def get_latest_financial_news(state: FinanceData):
    symbol = state.get("symbol", "AAPL")
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
    articles = requests.get(url).json()["articles"]
    headlines = [str(article.get("title", "")) for article in articles[:5]]
    summaries = [str(article.get("description", "")) for article in articles[:5]]
    record = {
        "timestamp": str(datetime.datetime.now()),
        "action": "get_latest_financial_news",
        "symbol": symbol,
        "headlines": headlines
    }
    news_col.insert_one(record)
    return to_serializable({"news": NewsState(headlines=headlines, summaries=summaries)})

def get_sec_filings(state: FinanceData) -> List[str]:
    ticker = state.get("symbol", "AAPL")
    form_types = ["10-K", "10-Q", "8-K"]
    size = 5
    url = f"https://api.sec-api.io"
    query = {
        "query": {
            "query_string": {
                "query": f"ticker:{ticker} AND formType:({' OR '.join(form_types)})"
            }
        },
        "from": "0",
        "size": str(size),
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    response = requests.post(
        url,
        headers={"Authorization": SEC_EDGAR_API_KEY, "Content-Type": "application/json"},
        json=query
    )
    data = response.json()
    filings = [
        {
            "formType": f["formType"],
            "filedAt": f["filedAt"],
            "title": f.get("title"),
            "description": f.get("description"),
            "link": f["linkToFilingDetails"]
        }
        for f in data.get("filings", [])
    ]
    record = {
        "timestamp": str(datetime.datetime.now()),
        "action": "get_sec_filings",
        "symbol": ticker,
        "filings_count": len(filings)
    }
    filings_col.insert_one(record)
    return to_serializable({"filings": filings})

def get_feature_data(state: FinanceData) -> dict:
    symbol = state.get("symbol", "AAPL")
    data = yf.download(symbol, period="1y", interval="1d")

    if data.empty or "Close" not in data:
        raise ValueError(f"No market data found for {symbol}")

    close = data["Close"]

    # Moving averages
    ma50 = float(close.rolling(window=50, min_periods=1).mean().iloc[-1])
    ma200 = float(close.rolling(window=200, min_periods=1).mean().iloc[-1])

    # RSI calculation
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain_series = gain.rolling(window=14, min_periods=14).mean()
    avg_loss_series = loss.rolling(window=14, min_periods=14).mean()

    avg_gain = float(avg_gain_series.iloc[-1])
    avg_loss = float(avg_loss_series.iloc[-1])

    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    # MACD calculation
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_value = float(macd_line.iloc[-1] - signal_line.iloc[-1])

    # Volatility
    log_returns = np.log(close / close.shift(1))
    volatility = float(log_returns.rolling(window=30, min_periods=1).std().iloc[-1])

    # Risk score computation
    volatility_risk = min(volatility / 0.05, 1.0)

    if rsi > 70:
        rsi_risk = (rsi - 70) / 30
    elif rsi < 30:
        rsi_risk = (30 - rsi) / 30
    else:
        rsi_risk = 0.0

    macd_risk = min(abs(macd_value) / 2.0, 1.0)
    ma_risk = 1.0 if ma50 < ma200 else 0.0
    risk_score = 0.4 * volatility_risk + 0.25 * rsi_risk + 0.2 * macd_risk + 0.15 * ma_risk

    record = {
        "timestamp": str(datetime.datetime.now()),
        "action": "get_feature_data",
        "symbol": symbol,
        "risk_score": float(risk_score)
    }
    feature_data_col.insert_one(record)

    return to_serializable({
        "feature_data": FeatureData(
            symbol=symbol,
            moving_average_50=ma50,
            moving_average_200=ma200,
            rsi=rsi,
            macd=macd_value,
            volatility=volatility,
            risk_score=risk_score
        )
    })

def generate_signal(state: dict) -> dict:
    symbol = state.get("symbol", "AAPL")
    data = yf.download(symbol, period="2y", interval="1d")
    data.dropna(inplace=True)
    data["Return"] = data["Close"].pct_change()
    data["MA5"] = data["Close"].rolling(window=5).mean()
    data["MA20"] = data["Close"].rolling(window=20).mean()
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["Volatility"] = np.log(data["Close"] / data["Close"].shift(1)).rolling(window=30).std()
    data["NextReturn"] = data["Return"].shift(-1)
    data["Target"] = np.where(
        data["NextReturn"] > 0.003, 1, 
        np.where(data["NextReturn"] < -0.003, -1, 0)
    )
    data.dropna(inplace=True)
    features = data[["MA5", "MA20", "RSI", "MACD", "Volatility"]]
    target = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)
    test_preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, test_preds)
    latest_features = features.tail(1)
    scaled = scaler.transform(latest_features)
    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0].max()
    signal = "BUY" if prediction == 1 else "SELL" if prediction == -1 else "HOLD"
    record = {
        "timestamp": str(datetime.datetime.now()),
        "action": "generate_signal",
        "symbol": symbol,
        "signal": signal,
        "confidence": round(prob * 100, 2),
        "accuracy": round(acc * 100, 2)
    }
    signal_col.insert_one(record)
    return to_serializable({
        "symbol": symbol,
        "strategy": f"{signal} signal generated (confidence: {round(prob * 100, 2)}%)",
        "signal": signal,
        "confidence": round(prob * 100, 2),
        "accuracy": round(acc * 100, 2)
    })

def check_compliance(state: FinanceData):
    symbol = state.get("symbol", "AAPL")
    signal = state.get("signal", "HOLD")
    risk_score = state.get("feature_data", {}).risk_score if "feature_data" in state else None
    restricted_list = ["TSLA", "GME", "AMC"]
    signal_based_limit = {"BUY": 0.65, "SELL": 0.8, "HOLD": 1.0}
    if symbol in restricted_list:
        compliance_status = "REVIEW"
        reason = f"{symbol} is in the restricted list and needs manual review."
    elif risk_score and risk_score > signal_based_limit.get(signal, 0.7):
        compliance_status = "FAILED"
        reason = f"Signal '{signal}' exceeds risk threshold ({risk_score:.2f} > {signal_based_limit.get(signal, 0.7):.2f})."
    else:
        compliance_status = "PASS"
        reason = f"Signal '{signal}' passed compliance checks with risk score {risk_score:.2f}."
    record = {
        "timestamp": str(datetime.datetime.now()),
        "action": "check_compliance",
        "symbol": symbol,
        "signal": signal,
        "compliance": compliance_status,
        "risk_score": risk_score,
        "reason": reason
    }
    compliance_col.insert_one(record)
    return to_serializable({"compliance": compliance_status, "risk": f"{risk_score:.2f}" if risk_score else "N/A", "compliance_reason": reason})

def provide_financial_advise(state: FinanceData):
    context = f"""
    Symbol: {state.get("symbol", "AAPL")}
    Market Data: {state.get("market_data", "N/A")}
    News: {state.get("news", "N/A")}
    Filings: {state.get("filings", "N/A")}
    Feature Data: {state.get("feature_data", "N/A")}
    Signal: {state.get("signal", "N/A")}
    Confidence: {state.get("confidence", "N/A")}
    Accuracy: {state.get("accuracy", "N/A")}
    Compliance: {state.get("compliance", "N/A")}
    Risk: {state.get("risk", "N/A")}
    """
    prompt = f"""
        You are a licensed financial advisor providing professional investment insights. 
        Carefully analyze the following information, including technical indicators, key KPIs, market news, SEC filings, and feature data:

        {context}

        Craft a comprehensive analysis and actionable recommendation for the user. 
        Start your response with a phrase like: "Given the technical indicators and key metrics..." 
        and include a compliance-friendly disclaimer. Provide insights in a clear, engaging, and step-by-step manner.
    """
    response = llm.invoke(prompt)
    record = {
        "timestamp": str(datetime.datetime.now()),
        "action": "provide_financial_advise",
        "symbol": state.get("symbol", "AAPL"),
        "advice": response.content
    }
    advice_col.insert_one(record)
    return to_serializable({"advice": response.content})

def perform_human_review(state: FinanceData):
    record = {
        "timestamp": str(datetime.datetime.now()),
        "action": "perform_human_review",
        "symbol": state['market_data'].symbol
    }
    human_review_col.insert_one(record)
    return to_serializable({"advice": "Manual review required due to compliance."})

def audit(state: FinanceData):
    symbol = state.get("symbol", "N/A")
    compliance = state.get("compliance", "N/A")
    signal = state.get("signal", "N/A")
    risk = state.get("risk", "N/A")
    confidence = state.get("confidence", "N/A")
    accuracy = state.get("accuracy", "N/A")
    advice = state.get("advice", "N/A")
    reason = state.get("compliance_reason", "N/A")

    decision_source = "Automated Financial Advisory" if compliance == "PASS" else "Manual Compliance Review"

    audit_record = {
        "timestamp": str(datetime.datetime.now()),
        "symbol": symbol,
        "workflow_stage": "audit",
        "signal": signal,
        "confidence": confidence,
        "accuracy": accuracy,
        "risk": risk,
        "compliance_status": compliance,
        "compliance_reason": reason,
        "decision_source": decision_source,
        "final_advice": advice,
        "market_data": (
            state.get("market_data").dict()
            if isinstance(state.get("market_data"), BaseModel)
            else state.get("market_data", {})
        ),
        "feature_data": (
            state.get("feature_data").dict()
            if isinstance(state.get("feature_data"), BaseModel)
            else state.get("feature_data", {})
        ),
        "news_headlines": (
            state.get("news").headlines if isinstance(state.get("news"), BaseModel)
            else state.get("news", {}).get("headlines", [])
        ),
        "filings_count": len(state.get("filings", [])) if "filings" in state else 0,
    }

    audit_col.insert_one(audit_record)

    with open("audit_log.json", "a") as f:
        f.write(json.dumps(audit_record, indent=2, default=str) + "\n")

    print(f"Audit log recorded for {symbol} | Decision: {decision_source}")
    return to_serializable({"audit": "logged"})

def validate_financial_decision(state: FinanceData) -> Literal['financial advisory','human review']:
    if state.get('compliance') == 'PASS':
        return "financial advisory"
    else:
        return "human review"

graph = StateGraph(FinanceData)

# Add nodes to the graph
# Data Ingestion
graph.add_node("market data", get_market_data)
graph.add_node("news", get_latest_financial_news)
graph.add_node("sec filings", get_sec_filings)

# Feature Engineering + ML Signal Generation
graph.add_node("feature engineering", get_feature_data)
graph.add_node("signal generation", generate_signal)

# Compliance & Human Oversibght
graph.add_node("compliance check", check_compliance)
graph.add_node("human review", perform_human_review)

# Financial Advisory & Audit
graph.add_node("financial advisory", provide_financial_advise)
graph.add_node("audit", audit)

# Add edges to define relationships between nodes
graph.add_edge(START, "market data")
graph.add_edge(START, "news")
graph.add_edge(START, "sec filings")

graph.add_edge("market data", "feature engineering")
graph.add_edge("news", "feature engineering")
graph.add_edge("sec filings", "feature engineering")
graph.add_edge("feature engineering", "signal generation")
graph.add_edge("signal generation", "compliance check")

graph.add_conditional_edges(
    "compliance check",
    validate_financial_decision
)

graph.add_edge("financial advisory", "audit")
graph.add_edge("human review", "audit")

graph.add_edge("audit", END)

# Initialize in-memory checkpointing
memory = MemorySaver()

# Create a workflow
financial_workflow = graph.compile(checkpointer=memory)

# Visualize the graph
financial_workflow.get_graph().draw_png("financial_workflow.png")

# Invoke the workflow
final_result = financial_workflow.invoke(
    input={"symbol": "GOOGL"},
    config={"configurable": {"thread_id": "financial_workflow_run"}}
)