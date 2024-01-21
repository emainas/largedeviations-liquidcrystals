import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad 


class MeanField:
    def __init__(self, npy_file, num_particles, mean_chi, num_bins, D, rho):
        self.my_array = np.load(npy_file)
        self.N = num_particles
        self.chi = mean_chi
        self.b = num_bins
        self.D = D
        self.rho = rho

    def calculate_betaepsilon(self, low_limit, upper_limit, step):
        self.low_limit = low_limit
        self.upper_limit = upper_limit
        self.step = step
        K = self.upper_limit-self.low_limit
        M = 500
        tiny_number = 9*10^(-9)
        rho, bins, _ = plt.hist(self.my_array, bins=self.b, density=True)
        plt.clf()

        counter = np.zeros(K)
        raw_prob = np.zeros(M)
        probability = np.zeros(M)
        order_parameter = np.zeros(M)
        mean_value = np.zeros(M)

        file = open(f'results/{self.D}D_gay_berne/rho_0{self.rho}/N{self.N}/be.txt', "w")

        for a in range(low_limit, upper_limit, 1):
            
            be = a/step
            counter[a-low_limit] = be
            
            for i in range(M):
                s = i/(2*M)
                order_parameter[i] = s

                def order(x):
                    return np.sqrt(2/3)*np.sqrt(2*s*s + 2*x*x + 2*s*x)
                def pade(x):
                    return np.exp(-self.N*(-np.log1p(-order(x)) - order(x) + (-1/2 + 5/(2*self.chi))*order(x)**2 - (7/9 - 5/(3*self.chi) - be/3)*order(x)**3))
                def jacob(x):
                    return (s-x)*(2*s+x)*(s+2*x)

                temp_var = integrate.quad(lambda x: jacob(x)*pade(x), -s/2, s)[0]
                raw_prob[i] = temp_var

            probability = raw_prob / integrate.simps(raw_prob, order_parameter) 
            mean_value = order_parameter * probability

            I = integrate.simps(mean_value, order_parameter) 
            F = np.average((bins[:-1] + bins[1:]) / 2, weights=rho)
            file.write(f"be: {be}, <Q>_sim-<Q>_num: {abs(I - F)}\n")
            #print('be:', be, '<Q>_sim-<Q>_num:', abs(I - F))

            mean_value.fill(0)
            order_parameter.fill(0)
            raw_prob.fill(0)
            probability.fill(0)

        file.close()