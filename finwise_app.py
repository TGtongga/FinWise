import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import cvxpy as cp
from   scipy.optimize import minimize

# Initialize the app
apptitle = 'Risk Assessment & Portfolio Recommendation'
st.set_page_config(page_title=apptitle, page_icon=":bar_chart:", layout="wide")

st.markdown("""
    <style>
        .css-1d391kg {
            width: 400px;
        }
    </style>
""", unsafe_allow_html=True)

# Load questionnaire
question_path = "questionaire_data/questionaire_info.json"
with open(question_path, "r") as f:
    questionnaire = json.load(f)

# Load fund metadata
info_path = "fund_data/fund_info.json"
with open(info_path, "r") as f:
    fund_info = json.load(f)
fund_info_dict = {item["name"]: item for item in fund_info}

# Load fund data
data_folder = "fund_data"
fund_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
funds = [f.replace(".csv", "") for f in fund_files]
funds = [f for f in funds if f in fund_info_dict]

fund_data_dict = {}
for fund in funds:
    file_path = os.path.join(data_folder, f"{fund}.csv")
    df = pd.read_csv(file_path)
    try:
        df['NAV Date'] = pd.to_datetime(df['NAV Date'], format="%d %b %Y")
    except:
        df['NAV Date'] = pd.to_datetime(df['NAV Date'], format="%Y-%m-%d")
    df.sort_values(by='NAV Date', inplace=True)
    df = df[df['NAV Date'] >= datetime.today() - timedelta(days=365)]
    fund_data_dict[fund] = df

moderate_limit = 16
aggresive_limit = 26

# Title & Slogan
st.markdown("""
    <style>
    .full-width-title {
        font-size: 3em;
        font-weight: bold;
        color: #333333;
        text-align: center;
        margin-bottom: 10px;
        width: 100%;
        padding: 10px;
        border-radius: 8px;
    }
    .centered-slogan {
        font-size: 1.5em;
        color: #666666;
        text-align: center;
        margin-top: 0px;
        margin-bottom: 20px;
    }
    </style>
    <div class="full-width-title">FinWise®</div>
    <div class="centered-slogan">Smarter Investing Through Personalized Risk Assessment</div>
""", unsafe_allow_html=True)

st.markdown("""
    * This is an open-to-public web app that performs portfolio recommendation based on the user input.
    * The app is designed to help users understand their risk aptitude and make informed decisions about their investment portfolios.
""")

# Sidebar questionaire
st.sidebar.markdown("## Questionnaire")
username = st.sidebar.text_input("Enter your username")

answers = []
for idx, q in enumerate(questionnaire):
    st.sidebar.markdown(f"**{idx+1}. {q['type']}**")
    answers.append(
        st.sidebar.radio(q["question"], q["choices_list"], key=f"q{idx}", index=None)
    )

# Main Content
st.markdown("---")
st.markdown("## Assessment Result")

if username and all(a is not None for a in answers):
    total_score = 0
    for answer, question in zip(answers, questionnaire):
        score = question["risk_score_list"][question["choices_list"].index(answer)]
        total_score += score

    if 0 <= total_score <= moderate_limit:
        risk_level = "Conservative"
        description = "You have a low tolerance for risk and prefer safety and stability. You are uncomfortable with volatility and potential loss of capital, accepting lower returns to protect your investment. "
    elif moderate_limit + 1 <= total_score <= aggresive_limit:
        risk_level = "Moderate"
        description = "You have a moderate tolerance for risk. You're willing to accept some volatility and moderate losses in exchange for modest to moderate returns. "
    else:
        risk_level = "Aggressive"
        description = "You have a high tolerance for risk and are willing to endure significant fluctuations or losses for the chance of higher long-term returns. (This aligns with an aggressive investor profile focused on growth.)"

    if risk_level == "Conservative":
        color = "green"
    elif risk_level == "Moderate":
        color = "orange"
    else:
        color = "red"

    st.markdown(f"""
        <div style="font-size: 1.5em; font-weight: bold;">
            Hi, {username}, your risk appetite is 
            <span style="color: {color};">{risk_level}</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f"{description}")

    # Risk Score Gauge
    def get_color(score):
        if score <= moderate_limit:
            return "green"
        elif score <= aggresive_limit:
            return "orange"
        else:
            return "red"

    color_map = {
        "green": "#2ECC71",
        "orange": "#F39C12",
        "red": "#E74C3C"
    }

    gauge_color = get_color(total_score)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=total_score,
        title={'text': "Your Risk Score"},
        number={'font': {'color': color_map[gauge_color], 'size': 36}},
        gauge={
            'axis': {'range': [0, 35]},
            'bar': {'color': color_map[gauge_color]},
            'steps': [
                {'range': [0, moderate_limit], 'color': '#D5F5E3'},
                {'range': [moderate_limit, aggresive_limit], 'color': '#FCF3CF'},
                {'range': [aggresive_limit, 35], 'color': '#F5B7B1'}
            ],
            'threshold': {
                'line': {'color': color_map[gauge_color], 'width': 6},
                'thickness': 0.75,
                'value': total_score
            }
        }
    ))
    fig_gauge.update_layout(width=500, height=300, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")
    st.markdown("## Recommended Portfolio")

    selected_funds = funds[:10]

    def optimized_weights(risk_level):
        """
        Optimize portfolio weights based on risk level.
        """
        if risk_level == "Conservative":
            return [30.51, 0, 0, 0, 0, 0, 7.44, 0, 38.26, 23.79] # Optimized weights for Conservative risk level
        elif risk_level == "Moderate":
            return [19.45, 0, 0, 0, 0, 0, 0, 0, 42.66, 37.89] # Optimized weights for Moderate risk level
        else:
            return [0, 0, 0, 0, 0, 0, 0, 0, 40, 60] # Optimized weights for Aggressive risk level
    

    weights = optimized_weights(risk_level)

    st.markdown(f"""
        <div style="font-size: 1.5em; font-weight: bold;">
            Portfolio Allocation Based on 
            <span style="color: {color};">{risk_level}</span>
            Risk level
        </div>
    """, unsafe_allow_html=True)

    filtered_funds = [f for f, w in zip(selected_funds, weights) if w > 0]
    filtered_weights = [w for w in weights if w > 0]

    fig = go.Figure(data=[go.Pie(
        labels=filtered_funds,
        values=filtered_weights,
        hole=0.3,
        marker=dict(colors=px.colors.sequential.Plasma),
        hoverinfo="label+percent",
        textinfo='label+percent'
    )])
    fig.update_layout(width=700, height=700, margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ────────────────────────────────────────────────────────────────
    # Backtesting Performance for Past One Year
    # ────────────────────────────────────────────────────────────────
    def backtest_portfolio(selected_funds, weights_pct, fund_data_dict):
        """
        Build a price matrix for the last 365 days, calculate daily portfolio
        returns, and derive key performance statistics.
        """
        # 1. Build price DataFrame (index = NAV Date)
        price_df = None
        for fund in selected_funds:
            tmp = (
                fund_data_dict[fund][["NAV Date", "Nav Price (SGD)"]]
                .rename(columns={"Nav Price (SGD)": fund})
            )
            price_df = tmp if price_df is None else price_df.merge(tmp, on="NAV Date", how="outer")

        price_df.sort_values("NAV Date", inplace=True)
        price_df.set_index("NAV Date", inplace=True)
        price_df.ffill(inplace=True)  # forward-fill any gaps

        # 2. Convert displayed weights (percentage) to decimals
        w = np.array(weights_pct) / 100.0

        # 3. Daily simple returns matrix & portfolio return series
        daily_ret = price_df.pct_change().dropna()
        port_ret  = (daily_ret * w).sum(axis=1)

        # 4. Cumulative NAV series (start at 1)
        cum_nav = (1 + port_ret).cumprod()

        # 5. Performance statistics over the look-back window
        cum_return   = cum_nav.iloc[-1] - 1
        ann_std      = port_ret.std() * np.sqrt(252)
        ann_return   = port_ret.mean() * 252
        sharpe_ratio = ann_return / ann_std if ann_std != 0 else np.nan
        rolling_max  = cum_nav.cummax()
        max_dd       = (cum_nav / rolling_max - 1).min()

        stats = pd.DataFrame(
            {
                "Cumulative Return": [cum_return],
                "Annualised Std Dev": [ann_std],
                "Sharpe Ratio": [sharpe_ratio],
                "Max Drawdown": [max_dd],
            }
        ).T.rename(columns={0: "Value"})

        return stats, cum_nav

    # Run the back-test
    perf_table, cum_nav_series = backtest_portfolio(selected_funds, weights, fund_data_dict)

    # Show the performance table
    perf_display = perf_table.copy()           # 4 × 1  (index = metrics)

    for metric in perf_display.index:
        val = perf_display.loc[metric, "Value"]
        perf_display.loc[metric, "Value"] = (
            f"{val:.2f}" if metric == "Sharpe Ratio" else f"{val:.2%}"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
        <div style="font-size: 1.5em; font-weight: bold;">
            Backtesting Performance for Last 12 Months
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"***The yield may not be indicative of future performance and may vary depending on market conditions**")

    st.table(perf_display)


    # Plot cumulative return curve
    cum_fig = go.Figure()
    cum_fig.add_trace(
        go.Scatter(
            x=cum_nav_series.index,
            y=cum_nav_series.values - 1,  # plot as excess return
            mode="lines",
            line=dict(color=color, width=3),
            hovertemplate="Date: %{x|%d %b %Y}<br>Cum Return: %{y:.2%}<extra></extra>",
            name="Portfolio"
        )
    )
    cum_fig.update_layout(
        title="Portfolio Cumulative Return – Last 12 Months",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_white",
        hovermode="x unified",
        width=900,
        height=450,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(cum_fig, use_container_width=True)

    with st.expander("📌 **More Fund Details**"):
        selected_fund = st.selectbox("Choose a fund to view details:", selected_funds)
        info = fund_info_dict[selected_fund]
        data = fund_data_dict[selected_fund]

        st.markdown(f"#### Fund: {selected_fund}")
        st.markdown(f"**Type:** {info['type']}")
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**Risk Rating:** {info['risk_rating']}")
        st.markdown(f"**Asset Under Management (AUM):** {info['aum']}")
        st.markdown(f"**3 YR Sharpe Ratio:** {info['sharpe_ratio']}")

        hover_text = [
            f"Date: {row['NAV Date'].strftime('%d %b %Y')}<br>Price: {row['Nav Price (SGD)']} SGD<br>Type: {info['type']}<br>Description: {info['description']}"
            for _, row in data.iterrows()
        ]
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(
            x=data['NAV Date'],
            y=data['Nav Price (SGD)'],
            mode='lines+markers',
            line=dict(color=color, width=3),
            marker=dict(size=6),
            name=selected_fund,
            hoverinfo='text',
            text=hover_text
        ))
        line_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="NAV Price (SGD)",
            width=700,
            height=400,
            template='plotly_white',
            hovermode="x unified"
        )
        st.plotly_chart(line_fig, use_container_width=True)
else:
    st.warning("Please complete the questionnaire in the side-bar to check your risk assessment results.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; padding: 10px; color: #888888; font-size: small;">
        FinWise® © 2025 | Created to promote financial literacy and informed investing.<br>
        This tool is designed for educational purposes only. Please consult a licensed financial advisor for investment decisions.<br><br>
        Contact Us:<br>
        Liu Yang | National University of Singapore<br>
        Tong Zhongyi | National University of Singapore<br>
        Yu Xinhui | National University of Singapore<br>
        Zhang Liang | National University of Singapore<br>
        Zou Zeren | National University of Singapore<br>
    </div>
""", unsafe_allow_html=True)


