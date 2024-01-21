
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import quad 

class AnalyticalMethods:
    def __init__(self, num_particles, mean_chi, beta_epsilon):
        self.N = num_particles
        self.chi = mean_chi
        self.be = beta_epsilon
        self.a = (5*self.N)/(3*self.chi)
        self.f = self.chi/self.N
        self.corr = 1/2
        self.C = (self.N/4)*(2/self.chi - 1 - self.corr)
    

    def central_limit_3D(self, x):
        return np.sqrt(6*self.a/np.pi)*(np.exp(-6*self.a*x*x) - (1 - (9/2)*self.a*x*x)*(np.exp(-(3/2)*self.a*x*x)))
    
    def central_limit_2D(self, x):
        return (2/self.f)*x*np.exp((-1/self.f)*(x**2))



    def clt_rate_3D(self, x):
        return -(1/self.N)*np.log(self.central_limit_3D(self, x))    
    
    def clt_rate_2D(self, x):
        return -(1/self.N)*np.log(self.central_limit_2D(self, x))
    

    def large_deviations_3D(self, x):
        M = 500
        small_value = 1e-10 # WITH CARE WHEN ADDING SMALL PROBABILITIES IN LDT
        raw_prob = np.zeros(M)
        order_parameter = np.linspace(0, 0.5, M)
        
        def order(x, s):
            return np.sqrt(2/3)*np.sqrt(2*s*s + 2*x*x + 2*s*x)
        def pade(x):
            return np.exp(-self.N*(-np.log1p(-order(x, s)) - order(x, s) + (-1/2 + 5/(2*self.chi))*order(x, s)**2 - (7/9 - 5/(3*self.chi) - self.be/3)*order(x, s)**3))
        def jacob(x, s):
            return (s-x)*(2*s+x)*(s+2*x)
        
        for i in range(M):
            
            s = order_parameter[i]        
            temp = integrate.quad(lambda x: jacob(x, s)*pade(x), -s/2, s)[0]
            raw_prob[i] = temp

        
        probability = raw_prob / integrate.simps(raw_prob, order_parameter)
        rate_function = -(1/self.N)*np.log(probability + small_value)

        return order_parameter, probability, rate_function
    
    def large_deviations_2D(self, x):
        def pade_raw(x):
            return 2*np.pi*x*((1-x**2)**(self.N/2))*np.exp(-self.N*(1/self.chi - 1/2)*(x**2))*np.exp(self.C*(x**4))*np.exp((self.N/4)*self.be*(x**4))
        
        r = integrate.quad(lambda x: pade_raw(x), 0, 1, full_output=1)[0]

        def pade_2D(x):
            return (1/r)*pade_raw(x)
        
        #def pade_rate_2D(x):
        #    return -(1/self.N)*np.log(pade_2D(x))

        return pade_2D(x)