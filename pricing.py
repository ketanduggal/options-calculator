import numpy as np
from scipy.stats import norm


def black_scholes_intermediates(S, K, T, sigma, r, q=0.0):
    """
    Compute the Black-Scholes auxiliary parameters d1, d2 and the associated
    standard-normal CDF values. Shared by pricing, Greeks and the UI derivation
    so the displayed intermediate values always match the computed premium.

    Parameters:
    S : float - Spot price
    K : float - Strike price
    T : float - Time to maturity in years
    sigma : float - Volatility
    r : float - Risk-free interest rate
    q : float - Continuous dividend yield (default 0.0)

    Returns:
    tuple - (d1, d2, N(d1), N(d2), N(-d1), N(-d2), phi(d1)); all None if T <= 0
    """
    if T <= 0:
        return (None,) * 7

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return d1, d2, norm.cdf(d1), norm.cdf(d2), norm.cdf(-d1), norm.cdf(-d2), norm.pdf(d1)


def black_scholes_call(S, K, T, sigma, r, q=0.0):
    """
    Calculate European Call option premium using Black-Scholes formula with dividend yield.

    Parameters:
    S : float - Spot price
    K : float - Strike price
    T : float - Time to maturity in years
    sigma : float - Volatility
    r : float - Risk-free interest rate
    q : float - Continuous dividend yield (default 0.0)

    Returns:
    float - Call option premium
    """
    if T <= 0:
        return max(S - K, 0)

    _, _, n_d1, n_d2, _, _, _ = black_scholes_intermediates(S, K, T, sigma, r, q)
    return S * np.exp(-q * T) * n_d1 - K * np.exp(-r * T) * n_d2


def black_scholes_put(S, K, T, sigma, r, q=0.0):
    """
    Calculate European Put option premium using Black-Scholes formula with dividend yield.

    Parameters:
    S : float - Spot price
    K : float - Strike price
    T : float - Time to maturity in years
    sigma : float - Volatility
    r : float - Risk-free interest rate
    q : float - Continuous dividend yield (default 0.0)

    Returns:
    float - Put option premium
    """
    if T <= 0:
        return max(K - S, 0)

    _, _, _, _, n_neg_d1, n_neg_d2, _ = black_scholes_intermediates(S, K, T, sigma, r, q)
    return K * np.exp(-r * T) * n_neg_d2 - S * np.exp(-q * T) * n_neg_d1


def calculate_greeks(S, K, T, sigma, r, q=0.0):
    """
    Calculate option Greeks with dividend yield.

    Returns:
    dict - Dictionary containing Delta (call & put), Gamma, Vega, Theta (call & put)
    """
    if T <= 0:
        return {
            'delta_call': 1.0 if S > K else 0.0,
            'delta_put': -1.0 if S < K else 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta_call': 0.0,
            'theta_put': 0.0
        }

    _, _, n_d1, n_d2, n_neg_d1, n_neg_d2, phi_d1 = black_scholes_intermediates(S, K, T, sigma, r, q)

    # Delta
    delta_call = np.exp(-q * T) * n_d1
    delta_put = np.exp(-q * T) * (n_d1 - 1)

    # Gamma (same for call and put)
    gamma = np.exp(-q * T) * phi_d1 / (S * sigma * np.sqrt(T))

    # Vega (same for call and put, divided by 100 for 1% change)
    vega = S * np.exp(-q * T) * phi_d1 * np.sqrt(T) / 100

    # Theta
    theta_call = (
        -S * np.exp(-q * T) * phi_d1 * sigma / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * n_d2
        + q * S * np.exp(-q * T) * n_d1
    )

    theta_put = (
        -S * np.exp(-q * T) * phi_d1 * sigma / (2 * np.sqrt(T))
        + r * K * np.exp(-r * T) * n_neg_d2
        - q * S * np.exp(-q * T) * n_neg_d1
    )

    return {
        'delta_call': delta_call,
        'delta_put': delta_put,
        'gamma': gamma,
        'vega': vega,
        'theta_call': theta_call,
        'theta_put': theta_put
    }


def monte_carlo_option_pricing(S, K, T, sigma, r, q, n_simulations=10000, option_type='call'):
    """
    Price options using Monte Carlo simulation with Geometric Brownian Motion.

    Parameters:
    S : float - Spot price
    K : float - Strike price
    T : float - Time to maturity
    sigma : float - Volatility
    r : float - Risk-free rate
    q : float - Dividend yield
    n_simulations : int - Number of Monte Carlo paths
    option_type : str - 'call' or 'put'

    Returns:
    tuple - (final_price, running_averages)
    """
    np.random.seed(42)  # For reproducibility

    # Generate random standard normal variables
    Z = np.random.standard_normal(n_simulations)

    # Simulate terminal stock prices using GBM
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Calculate payoffs
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    # Discount payoffs to present value
    discounted_payoffs = np.exp(-r * T) * payoffs

    # Calculate running averages for convergence visualization
    running_averages = np.cumsum(discounted_payoffs) / np.arange(1, n_simulations + 1)

    # Final option price estimate
    option_price = np.mean(discounted_payoffs)

    return option_price, running_averages


def monte_carlo_prices(S, K, T, sigma, r, q, n=10000):
    """
    Price both call and put options from a single shared GBM path.

    Because the call and put payoffs are computed from the same simulated
    terminal prices, only one random draw and one path simulation is needed,
    avoiding the double simulation of two separate calls.

    Parameters:
    S, K, T, sigma, r, q : float - standard option pricing inputs
    n : int - Number of Monte Carlo paths (default 10000)

    Returns:
    tuple - (call_price, call_running, put_price, put_running)
    """
    np.random.seed(42)  # For reproducibility

    Z = np.random.standard_normal(n)
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    disc = np.exp(-r * T)
    call = disc * np.maximum(ST - K, 0)
    put = disc * np.maximum(K - ST, 0)
    running = np.arange(1, n + 1)

    return (
        np.mean(call), np.cumsum(call) / running,
        np.mean(put), np.cumsum(put) / running,
    )


def generate_pnl_surface(S, K, T, sigma, r, q, call_purchase, put_purchase, n_spot=10, n_vol=10):
    """
    Compute a vectorized P&L surface over a spot/volatility grid.

    Both the call and put surfaces are computed from two broadcast Black-Scholes
    evaluations (no scalar loop), which is ~100x faster for a 10x10 grid.

    Parameters:
    S, K, T, sigma, r, q : float - standard option pricing inputs
    call_purchase, put_purchase : float - prices paid for each option
    n_spot, n_vol : int - grid resolution (default 10 each)

    Returns:
    tuple - (spot_range, vol_range, call_pnl, put_pnl)
      spot_range, vol_range : 1D arrays of length n_spot / n_vol
      call_pnl, put_pnl    : (n_vol, n_spot) arrays of P&L
    """
    spot_range = np.linspace(S * 0.8, S * 1.2, n_spot)
    vol_range = np.linspace(sigma * 0.8, sigma * 1.2, n_vol)
    SPOT, VOL = np.meshgrid(spot_range, vol_range)

    call = black_scholes_call(SPOT, K, T, VOL, r, q)
    put = black_scholes_put(SPOT, K, T, VOL, r, q)

    return spot_range, vol_range, call - call_purchase, put - put_purchase


def calculate_pnl(calculated_price, purchase_price):
    """
    Calculate Profit and Loss.

    Parameters:
    calculated_price : float - The calculated option premium
    purchase_price : float - The purchase price paid for the option

    Returns:
    float - PnL (calculated_price - purchase_price)
    """
    return calculated_price - purchase_price


def implied_volatility_newton_raphson(S, K, T, r, q, market_price, option_type='call', max_iterations=100, tolerance=1e-5):
    """
    Calculate implied volatility using Newton-Raphson method.

    Parameters:
    S : float - Spot price
    K : float - Strike price
    T : float - Time to maturity
    r : float - Risk-free rate
    q : float - Dividend yield
    market_price : float - Observed market price
    option_type : str - 'call' or 'put'
    max_iterations : int - Maximum iterations
    tolerance : float - Convergence tolerance

    Returns:
    tuple - (implied_volatility, iterations, initial_guess) or (None, 0, initial_guess) if not converged
    """
    # Initial guess: use approximation formula
    sigma = np.sqrt(2 * np.pi / T) * (market_price / S)
    sigma = max(0.01, min(sigma, 3.0))  # Bound initial guess between 1% and 300%
    initial_guess = sigma

    for iteration in range(max_iterations):
        # Compute d1, d2 and vega once, deriving price and vega together.
        # vega_full is the un-divided vega (the /100 in calculate_greeks is
        # undone), equal to S * e^{-qT} * phi(d1) * sqrt(T).
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        phi_d1 = norm.pdf(d1)
        vega_full = S * np.exp(-q * T) * phi_d1 * np.sqrt(T)

        if option_type == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        # Check for convergence
        price_diff = price - market_price
        if abs(price_diff) < tolerance:
            return sigma, iteration + 1, initial_guess

        # Avoid division by zero
        if abs(vega_full) < 1e-10:
            return None, 0, initial_guess

        # Newton-Raphson update
        sigma = sigma - price_diff / vega_full

        # Keep sigma in reasonable bounds
        sigma = max(0.001, min(sigma, 5.0))

    # Did not converge
    return None, 0, initial_guess


def get_binomial_params(S, K, T, sigma, r, q, N):
    """
    Compute the Cox-Ross-Rubinstein binomial tree parameters.

    Returns:
    tuple - (dt, u, d, p)
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp((r - q) * dt) - d) / (u - d)
    return dt, u, d, p


def american_option_binomial_tree(S, K, T, sigma, r, q, N, option_type='call'):
    """
    Price American options using Cox-Ross-Rubinstein Binomial Tree.

    Parameters:
    S : float - Spot price
    K : float - Strike price
    T : float - Time to maturity
    sigma : float - Volatility
    r : float - Risk-free rate
    q : float - Dividend yield
    N : int - Number of time steps
    option_type : str - 'call' or 'put'

    Returns:
    float - American option price
    """
    dt, u, d, p = get_binomial_params(S, K, T, sigma, r, q, N)

    # Initialize option values at maturity (vectorized over the N+1 terminal nodes)
    i = np.arange(N + 1)
    terminal_prices = S * (u ** (N - i)) * (d ** i)
    if option_type == 'call':
        option_values = np.maximum(terminal_prices - K, 0)
    else:
        option_values = np.maximum(K - terminal_prices, 0)

    # Backward induction through the tree
    for step in range(N - 1, -1, -1):
        for i in range(step + 1):
            # Calculate asset price at this node
            asset_price = S * (u ** (step - i)) * (d ** i)

            # Continuation value (discounted expected value)
            continuation_value = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])

            # Intrinsic value (early exercise value)
            if option_type == 'call':
                intrinsic_value = max(asset_price - K, 0)
            else:
                intrinsic_value = max(K - asset_price, 0)

            # American option: take maximum of continuation and early exercise
            option_values[i] = max(continuation_value, intrinsic_value)

    return option_values[0]
