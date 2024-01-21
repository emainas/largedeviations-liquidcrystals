
import numpy as np
import matplotlib.pyplot as plt


class Visualization:
    def __init__(self, num_particles, D, rho):
        self.N = num_particles
        self.D = D
        self.rho = rho

    def timeseries_file(self, var_name, npy_file):
        self.my_array = np.load(npy_file)
        r = len(self.my_array)
        X = np.linspace(0, r-1, r)
        x_ticks = [0, 3000000, 6000000, 9000000, 12000000, 15000000]
        x_labels = [0, 3, 6, 9, 12, 15]
        plt.xticks(x_ticks, x_labels)
        plt.scatter(X, self.my_array, color='blue', s=2)
        plt.xlabel(r'$M \quad \times \quad 10^6$', fontsize='20')
        plt.ylabel(var_name, fontsize='20')
        plt.title(fr'${self.D}D$, $N={self.N}$, $\rho^*=0.{self.rho}$, $T^*=1$', fontsize = '25')
        plt.tick_params(labelsize=20)
        plt.show()

    
    def multiple_densities(self, num_bins, npy_file, color, label , xrmt, yrmt, color1, label1, xldt, yldt, color2, label2):
        self.xrmt = xrmt
        self.yrmt = yrmt
        self.xldt = xldt
        self.yldt = yldt
        self.my_array = np.load(npy_file)
        rho, bins, _ = plt.hist(self.my_array, bins=num_bins, density=True)
        plt.clf()
        h = (bins[1] - bins[0])
        Q = bins[:-1] + h / 2

        plt.plot(Q, rho, label=label, color=color, marker='+', markersize=10, linestyle='None')
        plt.plot(xrmt, yrmt, label=label1, color=color1, lw=2)
        plt.plot(xldt, yldt, label=label2, color=color2, lw=2)
        plt.xlabel(r'$s$', fontsize='20')
        plt.ylabel(r'$p(s)$', fontsize='25')
        plt.xlim(0,0.5)
        plt.ylim(0,25)
        plt.tick_params(labelsize=20)
        plt.title(fr'${self.D}D$, $N={self.N}$, $\rho^*=0.{self.rho}$, $T^*=1$', fontsize = '25')
        plt.legend(loc='upper right', prop={'size': 16})
        plt.show() 

    def multiple_rates(self, num_bins, npy_file, color, label , xrmt, yrmt, color1, label1, xldt, yldt, color2, label2):
        self.xrmt = xrmt
        self.yrmt = yrmt
        self.xldt = xldt
        self.yldt = yldt
        self.my_array = np.load(npy_file)
        rho, bins, _ = plt.hist(self.my_array, bins=num_bins, density=True)
        plt.clf()
        h = (bins[1] - bins[0])
        Q = bins[:-1] + h / 2
        
                # Split the dataset into 10 independent runs
        num_runs = 10
        run_size = len(self.my_array) // num_runs
        split_data = [self.my_array[i:i + run_size] for i in range(0, len(self.my_array), run_size)]

        # Calculate the mean for each independent run
        run_means = [np.mean(run) for run in split_data]

        # Calculate the overall mean and estimate the standard error
        overall_mean = np.mean(run_means)
        standard_error = np.std(run_means) / np.sqrt(num_runs)

        # Calculate the standard deviation for each bin
        bin_stddevs = []
        for i in range(num_bins):
            bin_data = rho[i] * split_data[i % num_runs]
            bin_stddevs.append(np.std(bin_data))

        # Decrease the range of the bin_stddevs by sqrt(num_runs)
        for i in range(num_bins):
            bin_stddevs[i] /= np.sqrt(num_runs)

        # Plot the histogram with error bars
        plt.bar(Q, rho, width=h, align='center', alpha=0.7)
        plt.errorbar(Q, rho, yerr=bin_stddevs, fmt='none', color='black', capsize=10, label='Error Bars')

        # Add labels and title
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Histogram with Error Bars')

        # Show the plot
        plt.legend()
        plt.show()

        """
        exact_rate = -np.log(rho) 
        rmt_rate = -np.log(yrmt) 
        ldt_rate = -np.log(yldt) 

        
        plt.plot(Q, exact_rate, color=color, marker='o', markersize=10, linestyle='None')
        plt.plot(Q, exact_rate, color=color, linestyle='None')
        #plt.plot(xrmt, rmt_rate, label=label1, color=color1, lw=3, linestyle='dashed')
        plt.plot(xldt, ldt_rate, label=label2, color=color2, lw=3)

        ax = plt.gca()  # Get the current axes
        line_width = 3
        ax.spines['top'].set_linewidth(line_width)
        ax.spines['bottom'].set_linewidth(line_width)
        ax.spines['left'].set_linewidth(line_width)
        ax.spines['right'].set_linewidth(line_width)
        """
        """
        #For N=800 plot ρ = 0.285
        plt.text(0.27, 0.0095, fr'${self.D}D$, $N = {self.N}$, $\rho^*=0.{self.rho}$', fontsize=20)
        plt.text(0.273, 0.0068, '\u26AB\u26AB\u26AB',  color='red', fontsize=18)
        plt.text(0.323, 0.0068, 'Simulation',  color='black', fontsize=18)
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.7), prop={'size': 18}, frameon=False) 
        plt.yticks([0.3, 0.4, 0.5]) 
        y_ticks = [-0.002, 0.002, 0.006, 0.010]
        y_labels = [-2, 2, 6, 10]
        plt.yticks(y_ticks, y_labels)
        plt.tick_params(which='major', length=8, width=2, direction='in')
        plt.xlim(0.2438, 0.5782)
        plt.ylim(-0.0031, 0.0111)
        """

        """
        #For N=2048 plot ρ = 0.30
        ax.text(0.045, 0.9, '(a)', transform=ax.transAxes, fontsize=25, weight='bold')
        plt.text(0.065, 0.00028, fr'${self.D}D$, $N = {self.N}$, $\rho^*=0.{self.rho}0$', fontsize=20)
        plt.text(0.0529, -0.0002, '\u26AB\u26AB\u26AB',  color='red', fontsize=18)
        plt.text(0.0659, -0.0002, 'Simulation',  color='black', fontsize=18)
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.7), prop={'size': 18}, frameon=False) 
        x_ticks = [0.06, 0.09, 0.12]
        x_labels = [0.06, 0.09, 0.12]
        plt.xticks(x_ticks, x_labels)
        y_ticks = [-0.0015, -0.001, -0.0005, 0, 0.0005]
        y_labels = [-1.5, -1, -0.5, 0, 0.5]
        plt.yticks(y_ticks, y_labels)
        plt.tick_params(which='major', length=8, width=2, direction='in')
        plt.xlim(0.046, 0.1325)
        plt.ylim(-0.0017, 0.00049)
        
        
        #For N=2048 plot ρ = 0.304
        ax.text(0.045, 0.9, '(b)', transform=ax.transAxes, fontsize=25, weight='bold')
        plt.text(0.0935, 0.00073, fr'${self.D}D$, $N = {self.N}$, $\rho^*=0.{self.rho}$', fontsize=20)
        plt.text(0.077, 0.0002, '\u26AB\u26AB\u26AB',  color='red', fontsize=18)
        plt.text(0.0935, 0.0002, 'Simulation',  color='black', fontsize=18)
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.7), prop={'size': 18}, frameon=False) 
        x_ticks = [0.08, 0.12, 0.16]
        x_labels = [0.08, 0.12, 0.16]
        plt.xticks(x_ticks, x_labels)
        y_ticks = [-0.001, -0.0005, 0, 0.0005]
        y_labels = [-1, -0.5, 0, 0.5]
        plt.yticks(y_ticks, y_labels)
        plt.tick_params(which='major', length=8, width=2, direction='in')
        plt.xlim(0.068, 0.1769)
        plt.ylim(-0.001468, 0.00096)
        

        """
        """
        plt.xlabel(r'$s$', fontsize='30')
        plt.ylabel(r'$-\ln{p(s)}$', fontsize='25') 
        #plt.ylabel(r'$-\frac{1}{N} \ln{p(s)} (\times 10^{-3})$', fontsize='25') 
        plt.tick_params(labelsize=20) 
        plt.xlim(0,1)
        plt.tight_layout()        
        plt.show()
        #print(matplotlib.matplotlib_fname())
        #plt.savefig(fr'./publication_final_graphs/{self.D}D_rho_0{self.rho}_rates.jpeg', dpi=600, bbox_inches='tight')"""