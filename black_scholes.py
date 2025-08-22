#hello 

import math
import numpy as np
from scipy.stats import norm
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

def d1(S,X,b,sigma):
    '''
    this tells us in standard deviations, how far the excise price is from the mean when adjusted for a lognormal distribution.
    '''
    return (np.log(S/X) + (b + 0.5 * sigma**2)*t) / (sigma * np.sqrt(t))

def d2(d1, sigma,t):
    '''
    d2 tells us how many standard deviations below the mean is the median value.
    
    '''
    return d1 - sigma * np.sqrt(t)


#now we will define an equation calculating the value of a european call  and a european put using the Black-Scholes model. Fig 18.6 of the book.

def european_call(S,b,r,t,X,sigma):

    d1_value = d1(S, X, r, sigma)
    d2_value = d2(d1_value, sigma, t)
    
    call_value = (S * math.exp((b - r) * t) * norm.cdf(d1_value)) - (X * math.exp(-r * t) * norm.cdf(d2_value))  #norm.cdf is the standard cumulative normal distribution function.
    
    return call_value

def european_put(S,b,r,t,X,sigma):
    
    d1_value = d1(S, X, r, sigma)
    d2_value = d2(d1_value, sigma, t)
    
    put_value = (X * math.exp(-r * t) * norm.cdf(-d2_value)) - (S * math.exp((b - r) * t) * norm.cdf(-d1_value))
    
    return put_value

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
# First for the calls
def delta_call(S, b, r, t, X, sigma):
        d1_value = d1(S, X, r, sigma)

        '''
        Delta is the rate of change of the option price with respect to the underlying asset price.
        It tells us how much the option price will change for a small change in the underlying asset price.
        '''
        return math.exp((b - r) * t) * norm.cdf(d1_value)

def gamma(S, b, r, t, X, sigma):
    d1_value = d1(S, X, r, sigma)
    '''
    Gamma is the rate of change of delta with respect to the underlying asset price.
    It tells us how much delta will change for a small change in the underlying asset price.
    '''
    return (math.exp((b - r) * t) * norm.pdf(d1_value)) / (S * sigma * math.sqrt(t))

def theta_call(S,b,r,t,sigma,X):
    d1_value = d1(S, X, r, sigma)
    d2_value = d2(d1_value, sigma, t)
    '''
    Theta is the rate of change of the option price with respect to time.
    It tells us how much the option price will change for a small change in time.
    '''
    yearly =  (-S * math.exp((b - r) * t) * norm.pdf(d1_value) * sigma / (2 * math.sqrt(t))) -(b-r)*S*math.exp((b-r)*t)*norm.cdf(d1_value) -r*X*math.exp(-r * t) * norm.cdf(d2_value)
    return yearly / 365  # Return the daily theta by dividing by 365


def vega(S, b, r, t, X, sigma):
    d1_value = d1(S, X, r, sigma)
    '''
    Vega is the rate of change of the option price with respect to the volatility.
    It tells us how much the option price will change for a small change in the volatility.
    '''
    return (S * math.exp((b - r) * t) * norm.pdf(d1_value) * math.sqrt(t)) /100 # Divide by 100 to express vega in terms of a 1% change in volatility
     
def rho_call(S, b, r, t, X, sigma):

    d1_value = d1(S, X, r, sigma)
    d2_value = d2(d1_value, sigma, t)
    '''
    Rho is the rate of change of the option price with respect to the interest rate.
    It tells us how much the option price will change for a small change in the interest rate.
    '''
    C = european_call(S, b, r, t, X, sigma)
    if b == 0:
        rho = -t * C
    else:
        d2_value = d2(d1_value, sigma,t)
        rho = t * X * math.exp(-r * t) * norm.cdf(d2_value)
    return rho / 100.0 
    
# Now for the puts
def delta_put(S, b, r, t, X, sigma):
    d1_value = d1(S, X, r, sigma)

    '''
    Delta for put options is the rate of change of the option price with respect to the underlying asset price.
    '''
    return math.exp((b - r) * t) * (norm.cdf(d1_value) - 1)


def theta_put(S, b, r, t, X, sigma):
    d1_value = d1(S, X, r, sigma)
    d2_value = d2(d1_value, sigma, t)
    '''
    Theta for put options is the rate of change of the option price with respect to time.
    '''
    yearly = (-S * math.exp((b - r) * t) * norm.pdf(d1_value) * sigma / (2 * math.sqrt(t))) + (b - r) * S * math.exp((b - r) * t) * norm.cdf(d1_value) + r * X * math.exp(-r * t) * norm.cdf(-d2_value)
    return yearly / 365  # Return the daily theta by dividing by 365

def rho_put(S, b, r, t, X, sigma):
    P = european_put(S, b, r, t, X, sigma)
    if b == 0:
        rho = -t * P
    else:
        d2_value = d2(S, X, t, r, sigma, b)
        rho = -t * X * math.exp(-r * t) * norm.cdf(-d2_value)
    return rho / 100.0





def greeks_call(S, b, r, t, X, sigma):
    return {
        'Delta': delta_call(S, b, r, t, X, sigma),
        'Gamma': gamma(S, b, r, t, X, sigma),
        'Theta': theta_call(S, b, r, t, X, sigma),
        'Vega' : vega(S, b, r, t, X, sigma),
        'Rho'  : rho_call(S, b, r, t, X, sigma),
    }

def greeks_put(S, b, r, t, X, sigma):
    return {
        'Delta': delta_put(S, b, r, t, X, sigma),
        'Gamma': gamma(S, b, r, t, X, sigma),
        'Theta': theta_put(S, b, r, t, X, sigma),
        'Vega' : vega(S, b, r, t, X, sigma),
        'Rho'  : rho_put(S, b, r, t, X, sigma),
    }

# ---------- example ----------
if __name__ == "__main__":
    S = 100
    X = 100
    t = 1.0
    r = 0.05
    sigma = 0.20
    b = r  # non-dividend stock; use b=r-q if dividend yield q

    C = european_call(S, b, r, t, X, sigma)
    P = european_put(S, b, r, t, X, sigma)
    Gc = greeks_call(S, b, r, t, X, sigma)
    Gp = greeks_put(S, b, r, t, X, sigma)

    print(f"Call: {C:.4f}  Put: {P:.4f}")
    print("Call Greeks:", Gc)
    print("Put  Greeks:", Gp)

    # Put–call parity sanity check
    lhs = C - P
    rhs = S * math.exp((b - r) * t) - X * math.exp(-r * t)
    assert abs(lhs - rhs) < 1e-10, "Put–Call parity failed"
    