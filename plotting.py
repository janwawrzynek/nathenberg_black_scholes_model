#plotting.py
import numpy as np
import matplotlib.pyplot as plt
import black_scholes 
import argparse     

class Plotting :
    def __init__(self, model: black_scholes.Blackscholesmodel):
        self.model = model

    def option_value_vs_time(self):
        """Plot the value of European call and put options as a function of time to expiration."""
        t_range = np.linspace(1, 1/365, 365)  # time to expiration in years , going to explore the price from an expiry 1 year away, down to 1 day away.
        C_time_list = []
        P_time_list = []
        S = 100.0; X = 100.0; r = 0.05; sigma = 0.20

        for times in t_range:

            
            b = r  # Black–Scholes (non-dividend). For Black (futures) use b=0; for FX use b=r-rf.

            C_value = black_scholes.Blackscholesmodel.european_call(S, b, r, times, X, sigma)
            C_time_list.append(C_value)
            P_value = black_scholes.Blackscholesmodel.european_put (S, b, r, times, X, sigma)
            P_time_list.append(P_value)



        plt.plot(t_range, C_time_list, label='Call Value')
        plt.plot(t_range, P_time_list, label='Put Value')
        plt.xlim(1,0)  # Set x-axis limits to show the range of time to expiration
        #plt.grid()
        plt.xlabel('Time to Expiration (Years)', fontsize=20)
        plt.ylabel('Option Value', fontsize=20)
        plt.title('European Call and Put Option Values as a function of Time to Expiration', fontsize=10)
        plt.legend(prop = {'size': 20})
        plt.show()

    def option_value_vs_spot(self):
        """Plot the value of European call and put options as a function of spot price."""
        S_range = np.linspace(50, 150, 100)
        C_spot_list = []
        P_spot_list = []
        X = 100.0; r = 0.05; sigma = 0.20; b = r
        for S in S_range:
            C_value = black_scholes.Blackscholesmodel.european_call(S, b, r, self.model.t, X, sigma)
            C_spot_list.append(C_value)
            P_value = black_scholes.Blackscholesmodel.european_put(S, b, r, self.model.t, X, sigma)
            P_spot_list.append(P_value)
        plt.plot(S_range, C_spot_list, label='Call Value')
        plt.plot(S_range, P_spot_list, label='Put Value')
        plt.xlabel('Spot Price', fontsize=20)
        plt.ylabel('Option Value', fontsize=20)
        plt.title('European Call and Put Option Values as a function of Spot Price', fontsize=10)
        plt.legend(prop = {'size': 20})
        plt.show()

            
            
    
def main():
    parser = argparse.ArgumentParser(description="Choose which Black–Scholes plot to show")
    parser.add_argument("plot", choices=["time", "spot", "all"],
                        help="Which plot to display")
    args = parser.parse_args()

    model = black_scholes.Blackscholesmodel(S=100.0, X=100.0, t=1.0, r=0.05, sigma=0.20)
    p = Plotting(model)

    if args.plot in ("time", "all"):
        p.option_value_vs_time()
    if args.plot in ("spot", "all"):
        p.option_value_vs_spot()





if __name__ == "__main__":
    main()

# To run the code , type into the terminal
# python plotting.py time
# python plotting.py spot
# python plotting.py all