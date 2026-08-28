import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from datetime import datetime
from pricing import black_scholes_call, black_scholes_put, calculate_pnl, calculate_greeks, implied_volatility_newton_raphson, american_option_binomial_tree, black_scholes_intermediates, generate_pnl_surface, monte_carlo_prices, monte_carlo_option_pricing, get_binomial_params
from database import initialize_database, save_calculation
from pdf_report import generate_term_sheet

# Initialize database
initialize_database()

# Cache wrappers to avoid heavy recompute on UI reruns (Pure math is kept in pricing.py)
@st.cache_data
def cached_monte_carlo(S, K, T, sigma, r, q, n=10000):
    return monte_carlo_prices(S, K, T, sigma, r, q, n)

@st.cache_data
def cached_binomial_tree(S, K, T, sigma, r, q, N):
    call = american_option_binomial_tree(S, K, T, sigma, r, q, N, 'call')
    put = american_option_binomial_tree(S, K, T, sigma, r, q, N, 'put')
    return call, put

@st.cache_data
def cached_pnl_surface(S, K, T, sigma, r, q, call_purchase, put_purchase):
    return generate_pnl_surface(S, K, T, sigma, r, q, call_purchase, put_purchase)

@st.cache_data(max_entries=20)
def cached_term_sheet(inputs, premiums, greeks, call_matrix, put_matrix, spot_range, vol_range):
    call_heatmap_fig = create_pnl_heatmap(call_matrix, spot_range, vol_range, "Call")
    put_heatmap_fig = create_pnl_heatmap(put_matrix, spot_range, vol_range, "Put")
    return generate_term_sheet(
        inputs, premiums, greeks, call_heatmap_fig, put_heatmap_fig
    ).getvalue()

def metric_card(label, value, kind="metric", background=None):
    card_cls = "highlight-metric" if kind == "metric" else "greeks-highlight"
    sub = "metric-label" if kind == "metric" else "greek-label"
    big = "metric-value" if kind == "metric" else "greek-value"
    bg = f' style="background: {background};"' if background else ""
    return f'<div class="{card_cls}"{bg}><div class="{sub}">{label}</div><div class="{big}">{value}</div></div>'

def pnl_background(pnl):
    return "linear-gradient(135deg, #2D6A4F 0%, #1B4332 100%)" if pnl >= 0 else "linear-gradient(135deg, #A30000 0%, #7F0000 100%)"

def create_pnl_heatmap(matrix, spot, vol, label):
    fig = px.imshow(matrix, x=spot, y=vol, text_auto=".2f", aspect="auto",
                    color_continuous_scale="viridis",
                    labels={"x": "Spot Price", "y": "Volatility", "color": "PnL (₹)"})
    fig.update_layout(title=f"{label} Option PnL", height=500,
                      margin=dict(l=70, r=20, t=60, b=70),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    # Inline contrasted cell label logic
    normalized = (matrix - matrix.min()) / (matrix.max() - matrix.min() or 1)
    annotations = []
    for row_index, y_val in enumerate(vol):
        for col_index, x_val in enumerate(spot):
            annotations.append({
                "x": x_val,
                "y": y_val,
                "text": f"{matrix[row_index, col_index]:.2f}",
                "showarrow": False,
                "font": {
                    "color": "#111827" if normalized[row_index, col_index] > 0.62 else "white",
                    "size": 10,
                },
            })
    fig.update_layout(annotations=annotations)
    return fig

def render_pnl_heatmap(matrix, spot, vol, label):
    fig = create_pnl_heatmap(matrix, spot, vol, label)
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

def render_convergence(running, premium, mc_price, header):
    steps = np.arange(100, len(running)+1, 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=running[steps-1], mode='lines',
                             name='Monte Carlo Estimate', line=dict(color='#52B788', width=2)))
    fig.add_trace(go.Scatter(x=steps, y=[premium]*len(steps), mode='lines',
                             name='Black-Scholes Price', line=dict(color='#2D6A4F', width=2, dash='dash')))
    fig.update_layout(title=header, xaxis_title="Number of Simulations",
                      yaxis_title="Option Price (₹)", height=400,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(x=0.7, y=0.98), margin=dict(l=70, r=20, t=60, b=70))
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    st.markdown(f"**Monte Carlo Estimate:** ₹{mc_price:.4f}")
    st.markdown(f"**Analytical Price:** ₹{premium:.4f}")
    st.markdown(f"**Difference:** ₹{abs(mc_price - premium):.4f}")

# Page configuration
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

if "calculation_history" not in st.session_state:
    st.session_state["calculation_history"] = []

# Custom CSS for professional styling with Montserrat and Green Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&display=swap');

    body, button, input, textarea, select {
        font-family: 'Montserrat', sans-serif !important;
    }

    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
        font-size: 1.25rem !important;
        line-height: 1 !important;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #F8FAF9 0%, #E8F5F0 100%);
    }

    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(27, 67, 50, 0.1);
        color: #1F2937 !important;
    }
    [data-testid="stMainBlockContainer"] p,
    [data-testid="stMainBlockContainer"] strong,
    [data-testid="stMainBlockContainer"] em {
        color: #1F2937 !important;
    }
    [data-testid="stSidebar"] {
        padding-top: 1rem;
        background: linear-gradient(180deg, #1B4332 0%, #2D6A4F 50%, #40916C 100%);
        box-shadow: 4px 0 20px rgba(27, 67, 50, 0.2);
    }
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        width: 21rem !important;
        min-width: 21rem !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        width: 21rem !important;
        min-width: 21rem !important;
        margin-left: 0 !important;
        transform: none !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stTextInput label {
        color: white !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    [data-testid="stSidebar"] input {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 8px !important;
        color: #1B4332 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    [data-testid="stSidebar"] input:focus {
        background: rgba(255, 255, 255, 1) !important;
        border-color: #95D5B2 !important;
        box-shadow: 0 0 0 3px rgba(149, 213, 178, 0.4) !important;
    }
    [data-testid="stSidebar"] input::placeholder {
        color: #6B7280 !important;
    }
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 30%, #52B788 70%, #74C69D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
    }
    .sub-header {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1B4332;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        letter-spacing: -0.5px;
        border-bottom: 3px solid #52B788;
        padding-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.15rem;
        font-weight: 500;
        color: #40916C;
        margin-bottom: 0.5rem;
        letter-spacing: 0.3px;
    }
    .author-credit {
        font-size: 1rem;
        font-weight: 600;
        color: #2D6A4F;
        font-style: italic;
    }
    .stMarkdown .katex {
        font-size: 1.3em !important;
        font-weight: 400 !important;
        color: #1B4332 !important;
    }
    .stMarkdown .katex-display {
        background: transparent;
        color: #1B4332 !important;
        padding: 0.5rem 0;
        margin: 1rem 0;
        box-shadow: none;
        border: none;
        overflow-x: auto;
    }
    .stMarkdown .katex * {
        color: #1B4332 !important;
    }
    .stMarkdown p:has(> .katex:only-child) {
        text-align: center !important;
    }
    .stMarkdown p:has(> .katex:only-child) .katex {
        display: inline-block;
    }
    .highlight-metric {
        background: linear-gradient(135deg, #2D6A4F 0%, #40916C 100%);
        color: white;
        padding: 1.75rem;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(45, 106, 79, 0.3);
        margin: 0.5rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .highlight-metric:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 32px rgba(45, 106, 79, 0.4);
    }
    .highlight-metric .metric-label {
        font-size: 0.85rem;
        font-weight: 700;
        opacity: 0.95;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .highlight-metric .metric-value {
        font-size: 2.25rem;
        font-weight: 800;
    }
    .greeks-highlight {
        background: linear-gradient(135deg, #52B788 0%, #74C69D 100%);
        color: white;
        padding: 1.35rem;
        border-radius: 16px;
        box-shadow: 0 6px 16px rgba(82, 183, 136, 0.3);
        margin: 0.4rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .greeks-highlight:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(82, 183, 136, 0.4);
    }
    .greeks-highlight .greek-label {
        font-size: 0.75rem;
        font-weight: 700;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .greeks-highlight .greek-value {
        font-size: 1.6rem;
        font-weight: 800;
    }
    .stButton>button {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.85rem 2.5rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        background: linear-gradient(135deg, #52B788 0%, #74C69D 100%) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(82, 183, 136, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(82, 183, 136, 0.4);
    }
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #95D5B2 50%, transparent 100%);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📈 Black-Scholes Options Pricer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Quantitative Finance Tool for European Options Pricing & Risk Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="author-credit">Built by Ketan Duggal, Senior Actuarial Analyst</div>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Input Parameters")

spot_price = st.sidebar.number_input("Spot Price (S)", min_value=0.01, value=100.0, step=0.01)
strike_price = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=0.01)
time_to_maturity = st.sidebar.number_input("Time to Maturity (T) in years", min_value=0.01, value=1.0, step=0.01)
volatility = st.sidebar.number_input("Volatility (σ)", min_value=0.0, value=0.2, step=0.01)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05, step=0.01)
dividend_yield = st.sidebar.number_input("Continuous Dividend Yield (q)", min_value=0.0, value=0.00, step=0.01)
binomial_steps = st.sidebar.number_input("Binomial Tree Steps (N)", min_value=10, max_value=1000, value=100, step=1)

st.sidebar.markdown("---")
st.sidebar.header("Purchase Prices")

call_purchase_price = st.sidebar.number_input("Call Purchase Price", min_value=0.0, value=10.0, step=0.01)
put_purchase_price = st.sidebar.number_input("Put Purchase Price", min_value=0.0, value=10.0, step=0.01)

calculate_button = st.sidebar.button("Calculate & Save", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.1) 100%); border-radius: 12px; border: 2px solid rgba(255,255,255,0.3);">
        <a href="https://www.linkedin.com/in/ketanduggal/" target="_blank" style="text-decoration: none; color: #95D5B2; font-weight: bold; font-size: 1.15rem;">
            🔗 Connect on LinkedIn
        </a>
        <div style="margin-top: 0.75rem; color: #E8F5F0; font-size: 0.95rem;">
            <strong>Ketan Duggal</strong><br>
            Senior Actuarial Analyst
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Main panel calculations
if calculate_button or 'calculated' not in st.session_state:
    call_premium = black_scholes_call(spot_price, strike_price, time_to_maturity, volatility, risk_free_rate, dividend_yield)
    put_premium = black_scholes_put(spot_price, strike_price, time_to_maturity, volatility, risk_free_rate, dividend_yield)

    st.session_state['calculated'] = True
    st.session_state['call_premium'] = call_premium
    st.session_state['put_premium'] = put_premium
    st.session_state['dividend_yield'] = dividend_yield

    spot_range, vol_range, call_pnl_matrix, put_pnl_matrix = cached_pnl_surface(
        spot_price, strike_price, time_to_maturity, volatility, risk_free_rate, dividend_yield,
        call_purchase_price, put_purchase_price
    )

    outputs_data = [(s, v, call_pnl_matrix[i, j], put_pnl_matrix[i, j])
                    for i, v in enumerate(vol_range) for j, s in enumerate(spot_range)]

    st.session_state['call_pnl_matrix'] = call_pnl_matrix
    st.session_state['put_pnl_matrix'] = put_pnl_matrix
    st.session_state['spot_range'] = spot_range
    st.session_state['vol_range'] = vol_range

    if calculate_button:
        calculation_id = save_calculation(
            spot_price, strike_price, time_to_maturity, volatility, risk_free_rate,
            call_purchase_price, put_purchase_price, outputs_data
        )
        st.session_state["calculation_history"].insert(0, {
            "calculation_id": calculation_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "spot_price": spot_price,
            "strike_price": strike_price,
            "time_to_maturity": time_to_maturity,
            "volatility": volatility,
            "risk_free_rate": risk_free_rate,
            "dividend_yield": dividend_yield,
            "call_purchase_price": call_purchase_price,
            "put_purchase_price": put_purchase_price,
        })
        st.sidebar.success(f"Calculation saved! ID: {calculation_id[:8]}...")

dividend_yield = st.session_state.get('dividend_yield', 0.0)
call_premium = st.session_state['call_premium']
put_premium = st.session_state['put_premium']
greeks = calculate_greeks(
    spot_price, strike_price, time_to_maturity, volatility, risk_free_rate, dividend_yield
)

term_sheet_signature = (
    spot_price, strike_price, time_to_maturity, volatility,
    risk_free_rate, dividend_yield, call_purchase_price, put_purchase_price,
)
term_sheet_pdf = cached_term_sheet(
    {
        "spot_price": spot_price,
        "strike_price": strike_price,
        "time_to_maturity": time_to_maturity,
        "volatility": volatility,
        "risk_free_rate": risk_free_rate,
        "dividend_yield": dividend_yield,
    },
    {"call": call_premium, "put": put_premium},
    greeks,
    st.session_state['call_pnl_matrix'],
    st.session_state['put_pnl_matrix'],
    st.session_state['spot_range'],
    st.session_state['vol_range'],
)
st.sidebar.download_button(
    "📄 Download Pricing Term Sheet (PDF)",
    data=term_sheet_pdf,
    file_name="Black_Scholes_Term_Sheet.pdf",
    mime="application/pdf",
    type="primary",
)

# ----------------- 1. Header & Dashboard -----------------
st.markdown('<div class="sub-header">🎯 Current Option Premiums & Greeks</div>', unsafe_allow_html=True)

call_pnl = calculate_pnl(call_premium, call_purchase_price)
put_pnl = calculate_pnl(put_premium, put_purchase_price)

col1, col2, col3, col4, col5 = st.columns(5)

col1.markdown(metric_card("Call Premium", f"₹{call_premium:.2f}"), unsafe_allow_html=True)
col1.markdown(metric_card("Call PnL", f"₹{call_pnl:.2f}", background=pnl_background(call_pnl)), unsafe_allow_html=True)

col2.markdown(metric_card("Put Premium", f"₹{put_premium:.2f}"), unsafe_allow_html=True)
col2.markdown(metric_card("Put PnL", f"₹{put_pnl:.2f}", background=pnl_background(put_pnl)), unsafe_allow_html=True)

col3.markdown(metric_card("Δ Call", f"{greeks['delta_call']:.4f}", kind="greek"), unsafe_allow_html=True)
col3.markdown(metric_card("Δ Put", f"{greeks['delta_put']:.4f}", kind="greek"), unsafe_allow_html=True)

col4.markdown(metric_card("Γ (Gamma)", f"{greeks['gamma']:.4f}", kind="greek"), unsafe_allow_html=True)
col4.markdown(metric_card("ν (Vega)", f"₹{greeks['vega']:.4f}", kind="greek"), unsafe_allow_html=True)

col5.markdown(metric_card("Θ Call", f"₹{greeks['theta_call']:.4f}", kind="greek"), unsafe_allow_html=True)
col5.markdown(metric_card("Θ Put", f"₹{greeks['theta_put']:.4f}", kind="greek"), unsafe_allow_html=True)

st.markdown("---")

# ----------------- 2. Section 1: Mathematical Derivation & Theoretical Valuation (Black-Scholes-Merton) -----------------
st.markdown('<div class="sub-header">📐 Mathematical Derivation & Theoretical Valuation</div>', unsafe_allow_html=True)
st.markdown("The theoretical pricing of European options is computed within the continuous-time framework of the **Generalized Black-Scholes-Merton model** with dividend yield.")
st.markdown("**Model variables**")
st.markdown("**$S_t$** is the spot price, **$K$** is the strike price, **$r$** is the risk-free rate, **$q$** is the continuous dividend yield, **$\\sigma$** is the implied volatility, and **$T-t$** is the time to maturity.")

st.markdown("The primary valuation equations for European Call and Put options with dividend yield are:")
st.markdown(r"$$C(S,t) = S_t e^{-q(T-t)} \cdot N(d_1) - K e^{-r(T-t)} \cdot N(d_2)$$")
st.markdown(r"$$P(S,t) = K e^{-r(T-t)} \cdot N(-d_2) - S_t e^{-q(T-t)} \cdot N(-d_1)$$")

st.markdown("First, compute the auxiliary parameters $d_1$ and $d_2$ (adjusted for dividend yield):")
st.markdown(r"$$d_1 = \frac{\ln(S_t/K) + (r - q + \frac{1}{2}\sigma^2)(T - t)}{\sigma\sqrt{T - t}}$$")
st.markdown(r"$$d_2 = d_1 - \sigma\sqrt{T - t}$$")

d1_call = (np.log(spot_price / strike_price) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
d2_call = d1_call - volatility * np.sqrt(time_to_maturity)
n_d1 = norm.cdf(d1_call)
n_d2 = norm.cdf(d2_call)

st.markdown(f"Substituting the market inputs yields $d_1 = {d1_call:.6f}$ and $d_2 = {d2_call:.6f}.$")
st.markdown(f"Evaluating the standard normal CDF gives $N(d_1) = {n_d1:.6f}$ and $N(d_2) = {n_d2:.6f}.$")

st.markdown("Finally, the theoretical option premiums are:")
st.markdown(f"$$C = ₹{call_premium:.4f}$$")
st.markdown(f"$$P = ₹{put_premium:.4f}$$")

st.markdown("---")

# ----------------- 3. Section 2: Implied Volatility Solver (Newton-Raphson Method) -----------------
st.markdown('<div class="sub-header">🔍 Implied Volatility Solver (Newton-Raphson Method)</div>', unsafe_allow_html=True)
st.markdown('<p style="color: #6B7280; font-size: 1rem; margin-bottom: 1.5rem;">Newton-Raphson iterative method to back-solve implied volatility from observed market prices.</p>', unsafe_allow_html=True)

iv_col1, iv_col2 = st.columns(2)

with iv_col1:
    iv_option_type = st.selectbox("Option Type", ["Call", "Put"], key="iv_option_type")

with iv_col2:
    default_market_price = call_premium if iv_option_type == "Call" else put_premium
    iv_market_price = st.number_input(f"Target Market Price (₹)", min_value=0.01, value=float(default_market_price), step=0.01, key="iv_market_price")

# Calculate Implied Volatility
iv_result, iv_iterations, iv_initial_guess = implied_volatility_newton_raphson(
    spot_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield,
    iv_market_price, option_type=iv_option_type.lower(), max_iterations=100, tolerance=1e-5
)

# Display result
if iv_result is not None:
    st.markdown(f"""
    <div class="highlight-metric" style="background: linear-gradient(135deg, #52B788 0%, #74C69D 100%); margin-top: 1rem; max-width: 400px;">
        <div class="metric-label">Implied Volatility (σ)</div>
        <div class="metric-value">{iv_result * 100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("⚠️ Implied Volatility calculation did not converge. Please check your inputs or try a different target price.")

# Detailed Mathematical Derivation
st.markdown("#### **Implied Volatility Mathematical Flow**")
st.markdown("The implied volatility solver uses the **Newton-Raphson numerical technique** to find the root of the pricing difference function. Let $f(\\sigma)$ be the difference between the theoretical option price and the target market price:")
st.markdown(r"$$f(\sigma) = C_{BS}(\sigma) - C_{Market} \quad \text{or} \quad f(\sigma) = P_{BS}(\sigma) - P_{Market}$$")
st.markdown("Taking the Taylor expansion, the iterative algorithm is defined as:")
st.markdown(r"$$\sigma_{n+1} = \sigma_n - \frac{f(\sigma_n)}{f'(\sigma_n)} = \sigma_n - \frac{C_{BS}(\sigma_n) - C_{Market}}{\mathcal{V}(\sigma_n)}$$")
st.markdown(r"Where the derivative of the option price with respect to volatility is **Vega ($\mathcal{V}$)**:")
st.markdown(r"$$\mathcal{V} = S_t e^{-q(T-t)} \sqrt{T-t} \cdot \phi(d_1)$$")
st.markdown(r"Here, $\phi(d_1)$ represents the standard normal probability density function (PDF):")
st.markdown(r"$$\phi(d_1) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}d_1^2}$$")

if iv_result is not None:
    st.markdown(f"*Iterating using initial volatility guess $\\sigma_0 = {iv_initial_guess:.4f}$, target price $= ₹{iv_market_price:.2f}$, convergence achieved in {iv_iterations} iterations with tolerance $< 10^{{-5}}$.*")

st.markdown("---")

# ----------------- 4. Section 3: American Option Pricing (Cox-Ross-Rubinstein Binomial Tree) -----------------
st.markdown('<div class="sub-header">🌳 American Option Pricing (Cox-Ross-Rubinstein Binomial Tree)</div>', unsafe_allow_html=True)
st.markdown('<p style="color: #6B7280; font-size: 1rem; margin-bottom: 1.5rem;">Cox-Ross-Rubinstein (CRR) Binomial Tree model with early exercise capability.</p>', unsafe_allow_html=True)

# Calculate American option prices
american_call_price, american_put_price = cached_binomial_tree(
    spot_price, strike_price, time_to_maturity, volatility, risk_free_rate, dividend_yield,
    int(binomial_steps)
)

am_col1, am_col2 = st.columns(2)

with am_col1:
    st.markdown(f"""
    <div class="highlight-metric" style="background: linear-gradient(135deg, #2D6A4F 0%, #40916C 100%);">
        <div class="metric-label">American Call Premium</div>
        <div class="metric-value">₹{american_call_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    call_premium_diff = american_call_price - call_premium
    st.markdown(f"""
    <div style="text-align: center; margin-top: 0.75rem; color: #2D6A4F; font-weight: 600;">
        Early Exercise Premium: ₹{call_premium_diff:.4f}
    </div>
    """, unsafe_allow_html=True)

with am_col2:
    st.markdown(f"""
    <div class="highlight-metric" style="background: linear-gradient(135deg, #2D6A4F 0%, #40916C 100%);">
        <div class="metric-label">American Put Premium</div>
        <div class="metric-value">₹{american_put_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    put_premium_diff = american_put_price - put_premium
    st.markdown(f"""
    <div style="text-align: center; margin-top: 0.75rem; color: #2D6A4F; font-weight: 600;">
        Early Exercise Premium: ₹{put_premium_diff:.4f}
    </div>
    """, unsafe_allow_html=True)

st.markdown('<p style="color: #2D6A4F; font-size: 0.95rem; margin-top: 1rem; font-style: italic; font-weight: 600;">American options may carry an early exercise premium, making them greater than or equal to their European counterparts.</p>', unsafe_allow_html=True)

# Binomial Tree Mathematics
dt = time_to_maturity / binomial_steps
u = np.exp(volatility * np.sqrt(dt))
d = np.exp(-volatility * np.sqrt(dt))
p_prob = (np.exp((risk_free_rate - dividend_yield) * dt) - d) / (u - d)

st.markdown("#### **Binomial Tree Mathematical Flow**")
st.markdown("The Cox-Ross-Rubinstein (CRR) discrete-time model approximates the continuous Geometric Brownian Motion by partitioning the time to maturity $T$ into $N$ discrete steps. The model parameters are formulated as follows:")
st.markdown("**Step Size & Movement Factors**")
st.markdown(r"$$\Delta t = \frac{T}{N}$$")
st.markdown(r"$$u = e^{\sigma\sqrt{\Delta t}} \quad \text{and} \quad d = e^{-\sigma\sqrt{\Delta t}} = \frac{1}{u}$$")
st.markdown("**Risk-Neutral Probability**")
st.markdown(r"$$p = \frac{e^{(r-q)\Delta t} - d}{u - d}$$")
st.markdown("**Backward Induction with Early Exercise**")
st.markdown("At maturity, option payoffs are initialized. Moving backwards step-by-step, the option value at each node $V_{i,j}$ is calculated as the maximum of the continuation value and the early exercise value:")
st.markdown(r"$$V_{i,j} = \max\left( \text{Intrinsic Payoff}, e^{-r\Delta t}(p V_{i+1, j+1} + (1-p) V_{i+1, j}) \right)$$")
st.markdown(f"*Computed using $N={int(binomial_steps)}$ steps with calculated parameters: $\\Delta t = {dt:.6f}$, $u = {u:.6f}$, $d = {d:.6f}$, $p = {p_prob:.6f}$.*")

st.markdown("---")

# ----------------- 5. Section 4: Sensitivity Analysis (Profit & Loss Heatmaps) -----------------
st.markdown('<div class="sub-header">📊 Sensitivity Analysis: Profit & Loss Surfaces</div>', unsafe_allow_html=True)
st.markdown('<p style="color: #6B7280; font-size: 1rem; margin-bottom: 2rem;">Two-dimensional visualization of portfolio P&L sensitivity to simultaneous changes in underlying spot price and implied volatility.</p>', unsafe_allow_html=True)

def add_contrasting_cell_labels(figure, matrix, x_values, y_values):
    normalized = (matrix - matrix.min()) / (matrix.max() - matrix.min() or 1)
    annotations = []
    for row_index, y_value in enumerate(y_values):
        for column_index, x_value in enumerate(x_values):
            annotations.append({
                "x": x_value,
                "y": y_value,
                "text": f"{matrix[row_index, column_index]:.2f}",
                "showarrow": False,
                "font": {
                    "color": "#111827" if normalized[row_index, column_index] > 0.62 else "white",
                    "size": 10,
                },
            })
    figure.update_traces(texttemplate="")
    figure.update_layout(annotations=annotations)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Call PnL Heatmap")
    render_pnl_heatmap(
        st.session_state['call_pnl_matrix'],
        st.session_state['spot_range'],
        st.session_state['vol_range'],
        "Call"
    )

with col2:
    st.subheader("Put PnL Heatmap")
    render_pnl_heatmap(
        st.session_state['put_pnl_matrix'],
        st.session_state['spot_range'],
        st.session_state['vol_range'],
        "Put"
    )

st.markdown("---")

# ----------------- 6. Section 5: Monte Carlo Simulation (Geometric Brownian Motion) -----------------
st.markdown('<div class="sub-header">🎲 Monte Carlo Simulation (Geometric Brownian Motion)</div>', unsafe_allow_html=True)
st.markdown('<p style="color: #6B7280; font-size: 1rem; margin-bottom: 2rem;">Demonstration of convergence to the theoretical Black-Scholes price using Monte Carlo simulation. The chart illustrates the Law of Large Numbers as simulated prices converge to analytical solutions.</p>', unsafe_allow_html=True)

# Run Monte Carlo simulations
mc_call_price, mc_call_running = monte_carlo_option_pricing(
    spot_price, strike_price, time_to_maturity, volatility, risk_free_rate, dividend_yield,
    n_simulations=10000, option_type='call'
)

mc_put_price, mc_put_running = monte_carlo_option_pricing(
    spot_price, strike_price, time_to_maturity, volatility, risk_free_rate, dividend_yield,
    n_simulations=10000, option_type='put'
)

# Create convergence visualization
simulation_steps = np.arange(100, 10001, 100)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Call Option Convergence")

    fig_mc_call = go.Figure()

    # Monte Carlo running average
    fig_mc_call.add_trace(go.Scatter(
        x=simulation_steps,
        y=mc_call_running[simulation_steps - 1],
        mode='lines',
        name='Monte Carlo Estimate',
        line=dict(color='#52B788', width=2)
    ))

    # Theoretical Black-Scholes price
    fig_mc_call.add_trace(go.Scatter(
        x=simulation_steps,
        y=[call_premium] * len(simulation_steps),
        mode='lines',
        name='Black-Scholes Price',
        line=dict(color='#2D6A4F', width=2, dash='dash')
    ))

    fig_mc_call.update_layout(
        title="Convergence to Theoretical Call Price",
        xaxis_title="Number of Simulations",
        yaxis_title="Option Price (₹)",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.7, y=0.98),
        margin=dict(l=70, r=20, t=60, b=70)
    )

    st.plotly_chart(fig_mc_call, width='stretch', config={'displayModeBar': False})
    st.markdown(f"**Monte Carlo Estimate:** ₹{mc_call_price:.4f}")
    st.markdown(f"**Analytical Price:** ₹{call_premium:.4f}")
    st.markdown(f"**Difference:** ₹{abs(mc_call_price - call_premium):.4f}")

with col2:
    st.subheader("Put Option Convergence")

    fig_mc_put = go.Figure()

    # Monte Carlo running average
    fig_mc_put.add_trace(go.Scatter(
        x=simulation_steps,
        y=mc_put_running[simulation_steps - 1],
        mode='lines',
        name='Monte Carlo Estimate',
        line=dict(color='#52B788', width=2)
    ))

    # Theoretical Black-Scholes price
    fig_mc_put.add_trace(go.Scatter(
        x=simulation_steps,
        y=[put_premium] * len(simulation_steps),
        mode='lines',
        name='Black-Scholes Price',
        line=dict(color='#2D6A4F', width=2, dash='dash')
    ))

    fig_mc_put.update_layout(
        title="Convergence to Theoretical Put Price",
        xaxis_title="Number of Simulations",
        yaxis_title="Option Price (₹)",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.7, y=0.98),
        margin=dict(l=70, r=20, t=60, b=70)
    )

    st.plotly_chart(fig_mc_put, width='stretch', config={'displayModeBar': False})
    st.markdown(f"**Monte Carlo Estimate:** ₹{mc_put_price:.4f}")
    st.markdown(f"**Analytical Price:** ₹{put_premium:.4f}")
    st.markdown(f"**Difference:** ₹{abs(mc_put_price - put_premium):.4f}")

st.markdown("---")

with st.expander("View Calculation History", expanded=False):
    history = st.session_state["calculation_history"][:5]
    if history:
        st.dataframe(history, width='stretch')
    else:
        st.info("No calculations saved yet.")
