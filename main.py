from workflow import financial_workflow, to_serializable, FeatureData, MarketData
import streamlit as st
import os, time, io
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from fpdf import FPDF

st.set_page_config(
    page_title="AI Financial Research & Advisory Agent",
    layout="wide",
    page_icon="💹"
)

st.title("💹 AI Financial Research & Advisory Agent")
st.write("Automated Financial Signal Generation, Compliance Validation, and Advisory Workflow")

# Sidebar for stock selection
st.sidebar.header("⚙️ Configuration")
symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL, GOOGL, MSFT)", "GOOGL")
run_button = st.sidebar.button("🚀 Generate Insights")

# Sidebar for stock price visualization
st.sidebar.header("📈 Stock Price Visualization")
period = st.sidebar.text_input("Period (e.g. 3mo, 6mo, 9mo, 1y, 2y, 5y, 10y, etc.)", "2y")
interval = st.sidebar.text_input("Interval (e.g. 1d, 1wk, 1mo)", "1d")
visualize_button = st.sidebar.button("📊 Visualize Stock Price")

st.sidebar.write("---")
if os.path.exists("financial_workflow.png"):
    st.sidebar.image("financial_workflow.png", caption="Workflow Structure", width='stretch')
else:
    st.sidebar.info("Run the app once to generate workflow visualization.")

if run_button:
    st.subheader(f"📊 Analyzing {symbol} ...")
    progress = st.progress(0)

    try:
        # Simulated progress
        for pct in range(0, 80, 10):
            time.sleep(0.3)
            progress.progress(pct)

        # Invoke workflow
        result = financial_workflow.invoke(
            input={"symbol": symbol},
            config={"configurable": {"thread_id": f"streamlit_run_{symbol}"}}
        )
        progress.progress(100)

        # Extract serialized result
        result_data = to_serializable(result)
        
        # Display Technical Indicators & Key Metrics
        st.markdown("### 🔹 Key Technical Indicators & Metrics")

        # Extract market data
        market_data = result_data.get("market_data", None)
        
        # Extract feature data
        feature_data = result_data.get("feature_data", None)

        if feature_data:
            # Convert BaseModel to dict
            if isinstance(feature_data, FeatureData):
                feature_dict = feature_data.model_dump()
            else:
                feature_dict = feature_data  
                
            if isinstance(market_data, MarketData):
                market_dict = market_data.model_dump()
            else:
                market_dict = market_data

            indicators = {
                "Last Close": market_dict.get("last_close", None),
                "Volume": market_dict.get("volume", None),
                "Open": market_dict.get("open", None),
                "Low": market_dict.get("low", None),
                "High": market_dict.get("high", None),
                "50-Day MA": feature_dict.get("moving_average_50", None),
                "200-Day MA": feature_dict.get("moving_average_200", None),
                "RSI": feature_dict.get("rsi", None),
                "MACD": feature_dict.get("macd", None),
                "Volatility": feature_dict.get("volatility", None),
                "Risk Score": feature_dict.get("risk_score", None)
            }
            
            for k, v in indicators.items():
                if v is None:
                    indicators[k] = float('nan')

            df_indicators = pd.DataFrame(indicators, index=[0]).T.rename(columns={0: "Value"})
            st.dataframe(df_indicators.style.format("{:.2f}"))
        else:
            st.info("No feature data available.")
        
        # Display Market News Highlightss
        st.markdown("### 📰 Recent News Headlines")
        news_state = result_data.get("news", {})
        
        if news_state:
            headlines = getattr(news_state, "headlines", [])
            summaries = getattr(news_state, "summaries", [])
        else:
            headlines = []
            summaries = []

        if headlines:
            for i, (h, s) in enumerate(zip(headlines, summaries), 1):
                st.markdown(f"**{i}. {h}**")
                st.markdown(f"*{s}*")
        else:
            st.info("No recent news available.")

        # Display SEC Filings
        st.markdown("### 📝 Recent SEC Filings")
        filings = result_data.get("filings", [])
        if filings:
            for f in filings:
                form_type = f.get("formType", "N/A")
                filed_at = f.get("filedAt", "N/A")
                link = f.get("link", "#")
                description = f.get("description", "No description provided.")
                st.markdown(f"[{form_type} - {filed_at}]({link}) : {description}")
        else:
            st.info("No filings available.")
                  
        # Display Final AI-Powered Advice
        st.markdown("### 💡 AI-Powered Financial Recommendation")
        advice = result_data.get("advice", "No advice generated.")
        st.markdown(
            f"<div style='padding:15px; border-radius:10px; background-color:#f0f9ff; "
            f"border-left:5px solid #1E90FF; font-size:16px; line-height:1.6;'>{advice}</div>",
            unsafe_allow_html=True
        )

        # Create PDF
        pdf_buffer = io.BytesIO()
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Add Unicode font
        pdf.add_font("DejaVu", "", "dejavu-sans/DejaVuSans.ttf", uni=True) 
        pdf.add_font("DejaVu", "B", "dejavu-sans/DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", "B", 16)

        # Title
        pdf.cell(0, 10, f"AI Financial Insights Report: {symbol}", ln=True, align="C")
        pdf.ln(10)

        # Technical Indicators
        pdf.set_font("DejaVu", "B", 14)
        pdf.cell(0, 8, "Key Technical Indicators & Metrics", ln=True)
        pdf.set_font("DejaVu", "", 12)
        for k, v in indicators.items():
            pdf.cell(0, 6, f"{k}: {v:.2f}", ln=True)
        pdf.ln(5)

        # Market News
        pdf.set_font("DejaVu", "B", 14)
        pdf.cell(0, 8, "Recent News Headlines", ln=True)
        pdf.set_font("DejaVu", "", 12)
        for i, (h, s) in enumerate(zip(headlines, summaries), 1):
            pdf.multi_cell(0, 6, f"{i}. {h}\n{s}\n")
        pdf.ln(5)

        # SEC Filings
        pdf.set_font("DejaVu", "B", 14)
        pdf.cell(0, 8, "Recent SEC Filings", ln=True)
        pdf.set_font("DejaVu", "", 12)
        for f in filings:
            pdf.multi_cell(
                0, 6,
                f"[{f.get('formType', 'N/A')} - {f.get('filedAt', 'N/A')}] : {f.get('description', 'No description')}\nLink: {f.get('link', '#')}\n"
            )
        pdf.ln(5)

        # AI-Powered Advice
        pdf.set_font("DejaVu", "B", 14)
        pdf.cell(0, 8, "AI-Powered Financial Recommendation", ln=True)
        pdf.set_font("DejaVu", "", 12)
        pdf.multi_cell(0, 6, advice)

        # Output to buffer
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)

        # Download button
        st.download_button(
            label="📥 Download Full Insights as PDF",
            data=pdf_buffer,
            file_name=f"{symbol}_Financial_Advice.pdf",
            mime="application/pdf"
        )
        
        st.markdown("---")
        st.markdown(
            """
            **How to interpret these insights:**  
            - Technical indicators show stock trends and risk metrics.  
            - News & filings provide context for company-specific and market events.  
            - AI advice combines all data to give a compliance-friendly recommendation.
            """
        )

    except Exception as e:
        st.error(f"❌ Workflow execution failed: {e}")

if visualize_button:
    # Download stock data
    data = yf.download(symbol, period=period, interval=interval)
    
    if not data.empty:
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(data.index, data["Close"], label="Close Price", color="#1E90FF")
        ax.plot(data.index, data["Close"].rolling(20).mean(), label="20-Day MA", linestyle="--")
        ax.plot(data.index, data["Close"].rolling(50).mean(), label="50-Day MA", linestyle="--")
        ax.set_title(f"{symbol} Stock Price ({period})", fontsize=16)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("No stock data available for selected period.")
        
st.caption("Built with ❤️ using LangGraph, LangChain, and Streamlit for actionable AI-driven financial insights.")