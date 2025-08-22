#hello 
#black_scholes.py
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
class Blackscholesmodel:
    def __init__(self, S, X, t, r, sigma, b=None):
        self.S = S  # Spot price
        self.X = X  # Exercise price
        self.t = t  # Time to expiration in years
        self.r = r  # Domestic interest rate
        self.sigma = sigma  # Annualized volatility
        self.b = b if b is not None else r  # Cost of carry (default to domestic interest rate)

    @staticmethod
    def d1(S, X, t, b, sigma):
        """Black–Scholes d1 with cost of carry b."""
        return (np.log(S / X) + (b + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    @staticmethod
    def d2(d1_value, sigma, t):
        return d1_value - sigma * np.sqrt(t)
    @staticmethod
    # ---------- prices ----------
    def european_call(S, b, r, t, X, sigma):
        d1_value = Blackscholesmodel.d1(S, X, t, b, sigma)
        d2_value = Blackscholesmodel.d2(d1_value, sigma, t)
        return S * math.exp((b - r) * t) * norm.cdf(d1_value) - X * math.exp(-r * t) * norm.cdf(d2_value)
    @staticmethod
    def european_put(S, b, r, t, X, sigma):
        d1_value = Blackscholesmodel.d1(S, X, t, b, sigma)
        d2_value = Blackscholesmodel.d2(d1_value, sigma, t)
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
    @staticmethod
    def delta_call(S, b, r, t, X, sigma):
        '''Delta is the rate of change of the option price with respect to the underlying asset price.'''
        d1_value = Blackscholesmodel.d1(S, X, t, b, sigma)
        return math.exp((b - r) * t) * norm.cdf(d1_value)
    @staticmethod
    def delta_put(S, b, r, t, X, sigma):
        d1_value = Blackscholesmodel.d1(S, X, t, b, sigma)
        return math.exp((b - r) * t) * (norm.cdf(d1_value) - 1.0)
    @staticmethod
    def gamma(S, b, r, t, X, sigma):
        '''Gamma is the rate of change of delta with respect to the underlying asset price.'''
        d1_value = Blackscholesmodel.d1(S, X, t, b, sigma)
        return math.exp((b - r) * t) * norm.pdf(d1_value) / (S * sigma * math.sqrt(t))
    @staticmethod
    def vega(S, b, r, t, X, sigma):
        '''Vega is the rate of change of the option price with respect to volatility.'''
        d1_value = Blackscholesmodel.d1(S, X, t, b, sigma)
        # per 1 vol point (1%) → divide by 100
        return (S * math.exp((b - r) * t) * norm.pdf(d1_value) * math.sqrt(t)) / 100.0
    @staticmethod
    def theta_call(S, b, r, t, X, sigma):
        '''Theta is the rate of change of the option price with respect to time.'''
        d1_value = Blackscholesmodel.d1(S, X, t, b, sigma)
        d2_value = Blackscholesmodel.d2(d1_value, sigma, t)
        yearly = (-(S * math.exp((b - r) * t) * norm.pdf(d1_value) * sigma) / (2 * math.sqrt(t))
                - (b - r) * S * math.exp((b - r) * t) * norm.cdf(d1_value)
                - r * X * math.exp(-r * t) * norm.cdf(d2_value))
        return yearly / 365.0  # per day
    @staticmethod
    def theta_put(S, b, r, t, X, sigma):
        d1_value = Blackscholesmodel.d1(S, X, t, b, sigma)
        d2_value = Blackscholesmodel.d2(d1_value, sigma, t)
        yearly = (-(S * math.exp((b - r) * t) * norm.pdf(d1_value) * sigma) / (2 * math.sqrt(t))
                + (b - r) * S * math.exp((b - r) * t) * norm.cdf(-d1_value)  # NOTE N(-d1)
                + r * X * math.exp(-r * t) * norm.cdf(-d2_value))
        return yearly / 365.0  # per day
    @staticmethod
    def rho_call(S, b, r, t, X, sigma):
        """Rho is the rate of change of the option price with respect to interest rate."""
        # Special futures-style case when b == 0 (Natenberg): rho = -t * price
        C = Blackscholesmodel.european_call(S, b, r, t, X, sigma)
        if b == 0:
            rho = -t * C
        else:
            d1_value = Blackscholesmodel.d1(S, X, t, b, sigma)
            d2_value = Blackscholesmodel.d2(d1_value, sigma, t)
            rho = t * X * math.exp(-r * t) * norm.cdf(d2_value)
        return rho / 100.0  # per 1% rate move
    @staticmethod
    def rho_put(S, b, r, t, X, sigma):
        P = Blackscholesmodel.european_put(S, b, r, t, X, sigma)
        if b == 0:
            rho = -t * P
        else:
            d1_value = Blackscholesmodel.d1(S, X, t, b, sigma)
            d2_value = Blackscholesmodel.d2(d1_value, sigma, t)
            rho = -t * X * math.exp(-r * t) * norm.cdf(-d2_value)
        return rho / 100.0  # per 1% rate move
    # Greeks for call and put options
    @staticmethod
    def greeks_call(S, b, r, t, X, sigma):
        d = {
            'Delta': Blackscholesmodel.delta_call(S, b, r, t, X, sigma),
            'Gamma': Blackscholesmodel.gamma(S, b, r, t, X, sigma),
            'Theta': Blackscholesmodel.theta_call(S, b, r, t, X, sigma),
            'Vega' : Blackscholesmodel.vega(S, b, r, t, X, sigma),
            'Rho'  : Blackscholesmodel.rho_call(S, b, r, t, X, sigma),
        }
        return {k: float(v) for k, v in d.items()} 
    @staticmethod
    def greeks_put(S, b, r, t, X, sigma):
        d = {
            'Delta': Blackscholesmodel.delta_put(S, b, r, t, X, sigma),
            'Gamma': Blackscholesmodel.gamma(S, b, r, t, X, sigma),
            'Theta': Blackscholesmodel.theta_put(S, b, r, t, X, sigma),
            'Vega' : Blackscholesmodel.vega(S, b, r, t, X, sigma),
            'Rho'  : Blackscholesmodel.rho_put(S, b, r, t, X, sigma),
        }
        return {k: float(v) for k, v in d.items()}



  

    