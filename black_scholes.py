#hello 

import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
'''
 My goal is to implement the Black-Scholes model for:
 1) Options for a non-dividend paying stock
 2) The Black Model for options on futures
 3) The Garman-Kohlhagen model for options on foreign currencies

 This work is based on the book by Sheldon Natenberg: Options Voltatility and Pricing, 2nd Edition.

'''

# Our variables are
#S the spot price or underlying asset price
# X = the exercise price
#t = time to expiration in years
#r = the domestic interest rate 
# sigma = the anualised volatility or standard deviation in percent

def d1(S, X, t, b, sigma):
    """Black–Scholes d1 with cost of carry b."""
    return (np.log(S / X) + (b + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))

def d2(d1_value, sigma, t):
    return d1_value - sigma * np.sqrt(t)

# ---------- prices ----------
def european_call(S, b, r, t, X, sigma):
    d1_value = d1(S, X, t, b, sigma)
    d2_value = d2(d1_value, sigma, t)
    return S * math.exp((b - r) * t) * norm.cdf(d1_value) - X * math.exp(-r * t) * norm.cdf(d2_value)

def european_put(S, b, r, t, X, sigma):
    d1_value = d1(S, X, t, b, sigma)
    d2_value = d2(d1_value, sigma, t)
    return X * math.exp(-r * t) * norm.cdf(-d2_value) - S * math.exp((b - r) * t) * norm.cdf(-d1_value)


'''
For the above European call and put options, we have a few different situations to consider:
if b =r : The Black-Scholes Model for options on a stock.
if b = r = 0 : The Black-Scholes Model for options on futures, where the options are subject to a futures type settlement.
if b = 0: The Black model for options on futures where the options are subject to stock-type settlement.
if b = r-r_f : The Garman-Kohlhagen model for options on foreign currencies, where r_f is the foreign interest rate.
For options on a dividend paying stock, the spot price,S, must be discounted by the value of the expected dividend payments.
This can be approximated by setting b = r = q, where q is the annual dividend yield in percent. 
For a more exact calculation we can deduct from S, the value of each dividend payment,D, together with the interest which can be earned on that dividened payment to expiration. 
S is then replaced by S- sum D_i e ^(-r * t_d) where t_d is the time to expiration of each dividend payment, to expiration of the option.
''' 

# Now I will define the Greeks, vega and gamma
def delta_call(S, b, r, t, X, sigma):
    '''Delta is the rate of change of the option price with respect to the underlying asset price.'''
    d1_value = d1(S, X, t, b, sigma)
    return math.exp((b - r) * t) * norm.cdf(d1_value)

def delta_put(S, b, r, t, X, sigma):
    d1_value = d1(S, X, t, b, sigma)
    return math.exp((b - r) * t) * (norm.cdf(d1_value) - 1.0)

def gamma(S, b, r, t, X, sigma):
    '''Gamma is the rate of change of delta with respect to the underlying asset price.'''
    d1_value = d1(S, X, t, b, sigma)
    return math.exp((b - r) * t) * norm.pdf(d1_value) / (S * sigma * math.sqrt(t))

def vega(S, b, r, t, X, sigma):
    '''Vega is the rate of change of the option price with respect to volatility.'''
    d1_value = d1(S, X, t, b, sigma)
    # per 1 vol point (1%) → divide by 100
    return (S * math.exp((b - r) * t) * norm.pdf(d1_value) * math.sqrt(t)) / 100.0

def theta_call(S, b, r, t, X, sigma):
    '''Theta is the rate of change of the option price with respect to time.'''
    d1_value = d1(S, X, t, b, sigma)
    d2_value = d2(d1_value, sigma, t)
    yearly = (-(S * math.exp((b - r) * t) * norm.pdf(d1_value) * sigma) / (2 * math.sqrt(t))
              - (b - r) * S * math.exp((b - r) * t) * norm.cdf(d1_value)
              - r * X * math.exp(-r * t) * norm.cdf(d2_value))
    return yearly / 365.0  # per day

def theta_put(S, b, r, t, X, sigma):
    d1_value = d1(S, X, t, b, sigma)
    d2_value = d2(d1_value, sigma, t)
    yearly = (-(S * math.exp((b - r) * t) * norm.pdf(d1_value) * sigma) / (2 * math.sqrt(t))
              + (b - r) * S * math.exp((b - r) * t) * norm.cdf(-d1_value)  # NOTE N(-d1)
              + r * X * math.exp(-r * t) * norm.cdf(-d2_value))
    return yearly / 365.0  # per day

def rho_call(S, b, r, t, X, sigma):
    """Rho is the rate of change of the option price with respect to interest rate."""
    # Special futures-style case when b == 0 (Natenberg): rho = -t * price
    C = european_call(S, b, r, t, X, sigma)
    if b == 0:
        rho = -t * C
    else:
        d1_value = d1(S, X, t, b, sigma)
        d2_value = d2(d1_value, sigma, t)
        rho = t * X * math.exp(-r * t) * norm.cdf(d2_value)
    return rho / 100.0  # per 1% rate move

def rho_put(S, b, r, t, X, sigma):
    P = european_put(S, b, r, t, X, sigma)
    if b == 0:
        rho = -t * P
    else:
        d1_value = d1(S, X, t, b, sigma)
        d2_value = d2(d1_value, sigma, t)
        rho = -t * X * math.exp(-r * t) * norm.cdf(-d2_value)
    return rho / 100.0  # per 1% rate move

def greeks_call(S, b, r, t, X, sigma):
    d = {
        'Delta': delta_call(S, b, r, t, X, sigma),
        'Gamma': gamma(S, b, r, t, X, sigma),
        'Theta': theta_call(S, b, r, t, X, sigma),
        'Vega' : vega(S, b, r, t, X, sigma),
        'Rho'  : rho_call(S, b, r, t, X, sigma),
    }
    return {k: float(v) for k, v in d.items()} 

def greeks_put(S, b, r, t, X, sigma):
    d = {
        'Delta': delta_put(S, b, r, t, X, sigma),
        'Gamma': gamma(S, b, r, t, X, sigma),
        'Theta': theta_put(S, b, r, t, X, sigma),
        'Vega' : vega(S, b, r, t, X, sigma),
        'Rho'  : rho_put(S, b, r, t, X, sigma),
    }
    return {k: float(v) for k, v in d.items()}

t_range = np.linspace(1, 1/365, 365)  # time to expiration in years , going to explore the price from an expiry 1 year away, down to 1 day away.
C_time_list = []
P_time_list = []

for times in t_range:

    S = 100.0; X = 100.0; r = 0.05; sigma = 0.20
    b = r  # Black–Scholes (non-dividend). For Black (futures) use b=0; for FX use b=r-rf.

    C_value = european_call(S, b, r, times, X, sigma)
    C_time_list.append(C_value)
    P_value = european_put(S, b, r, times, X, sigma)
    P_time_list.append(P_value)

plt.plot(t_range, C_time_list, label='Call Value')
plt.plot(t_range, P_time_list, label='Put Value')
plt.xlabel('Time to Expiration (Years)', fontsize=20)
plt.ylabel('Option Value', fontsize=20)
plt.title('European Call and Put Option Values as a function of Time to Expiration', fontsize=20)
plt.legend(prop = {'size': 20})
plt.show()

# ---------- example ----------
if __name__ == "__main__":
    S = 100.0; X = 100.0; t = 1.0; r = 0.05; sigma = 0.20
    b = r  # Black–Scholes (non-dividend). For Black (futures) use b=0; for FX use b=r-rf.

    C = european_call(S, b, r, t, X, sigma)
    P = european_put(S, b, r, t, X, sigma)
    Gc = greeks_call(S, b, r, t, X, sigma)
    Gp = greeks_put(S, b, r, t, X, sigma)

    print(f"Call Value: {C:.4f}  Put Value: {P:.4f}")
    print("Call Greeks:", Gc)
    print("Put  Greeks:", Gp)

    # Put–call parity sanity check
    lhs = C - P
    rhs = S * math.exp((b - r) * t) - X * math.exp(-r * t)
    assert abs(lhs - rhs) < 1e-10, "Put–Call parity failed"


  

    