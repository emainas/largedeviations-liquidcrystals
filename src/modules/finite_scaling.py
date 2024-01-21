
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('agg')

class FiniteScaling:
    def __init__(self, D, rho, N1, N2, N3, N4):
        self.D = D
        self.rho = rho
        num_graphs = 4
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.N4 = N4
        n_array = np.array([N1, N2, N3, N4])
        chi_array = np.zeros(num_graphs)
        be_array = np.zeros(num_graphs)
        for i in range(num_graphs):
            with open(f'results/{self.D}D_gay_berne/rho_0{self.rho}/N{n_array[i]}/mean_chi.txt', 'r') as f:
                chi_array[i] = float(f.readline().split()[0])

            with open(f'results/{self.D}D_gay_berne/rho_0{self.rho}/N{n_array[i]}/be.txt', 'r') as f:
                min_qsim_qnum = float("inf")
                be_value = None
                for line in f:
                    if line.startswith("be:"):
                        be_val, qsim_qnum = line.split(", ")
                        qsim_qnum_val = float(qsim_qnum.split(": ")[1])
                        if qsim_qnum_val < min_qsim_qnum:
                            min_qsim_qnum = qsim_qnum_val
                            be_value = float(be_val.split(": ")[1])
            be_array[i] = be_value

        self.N = n_array
        self.chi = chi_array
        self.be = be_array

    def finite_scaling_subplots(self):
        colors=['cyan', 'red', 'blue', 'black']
        labels=[str(self.N1), str(self.N2), str(self.N3), str(self.N4)]
        
        num_graphs = len(colors)
        num_bins = 39

        data_list = []
        x_rm_list = []
        x_ld_list = []
        y_rm_list = []
        y_ld_list = []

        rho_list = []
        bins_list = []
        bin_width_list = []
        bin_center_list = []

        for i in range(num_graphs):
            
            data_array = np.load(f'data/{self.D}D_gay_berne/rho_0{self.rho}/N{self.N[i]}/total_first.npy')
            data_list.append(data_array)

            x_rm_array = np.load(f'results/{self.D}D_gay_berne/rho_0{self.rho}/N{self.N[i]}/central_limit_{self.D}D_x.npy')
            x_rm_list.append(x_rm_array)
            y_rm_array = np.load(f'results/{self.D}D_gay_berne/rho_0{self.rho}/N{self.N[i]}/central_limit_{self.D}D_y.npy')
            y_rm_list.append(y_rm_array)
            
            x_ld_array = np.load(f'results/{self.D}D_gay_berne/rho_0{self.rho}/N{self.N[i]}/large_deviations_{self.D}D_x.npy')
            x_ld_list.append(x_ld_array)
            y_ld_array = np.load(f'results/{self.D}D_gay_berne/rho_0{self.rho}/N{self.N[i]}/large_deviations_{self.D}D_y.npy')
            y_ld_list.append(y_ld_array)
    
        for i in range(num_graphs):
            rho, bins, _ = plt.hist(data_list[i], bins=num_bins, density=True, color=colors[i], label=labels[i])
            rho_list.append(rho)
            bins_list.append(bins)
            bin_width_list.append(bins[1] - bins[0])
            bin_center_list.append((bins[:-1] + bins[1:]) / 2)
            plt.clf() 
        
        fig_width_in = 8.27
        fig_height_in = 11.69
        legend_fontsize_scale = 2.7
        tick_labelsize_scale = 4
        tick_labelpad_scale = 1.2
        xlabel_fontsize_scale = 5
        ylabel_fontsize_scale = 5
        y_labelpad_scale = +0.5
        x_labelpad_scale = -0.6
        text_scale = 3.5

        f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(fig_width_in, fig_height_in))
        f.subplots_adjust(hspace=0)
        if self.rho == '':
            for i in range(0, num_graphs, 1):
                if colors[i] == colors[-1]:
                    ax1.plot(x_rm_list[i], y_rm_list[i], color=colors[i], lw=4, label='CLT')
                    ax1.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None', label=fr'Simulation')
                    ax2.plot(x_ld_list[i], y_ld_list[i], color=colors[i], lw=4, label='LDT')
                    ax2.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None', label=fr'Simulation')
                else:
                    ax1.plot(x_rm_list[i], y_rm_list[i], color=colors[i], lw=4)
                    ax1.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None')
                    ax2.plot(x_ld_list[i], y_ld_list[i], color=colors[i], lw=4)
                    ax2.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None')
        else:
            if self.rho != 285:
                for i in range(0, num_graphs, 1):
                    if colors[i] == colors[-1]:
                        ax1.plot(x_rm_list[i], y_rm_list[i], color=colors[i], lw=4, label='CLT', linestyle='dashed')
                        ax1.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None')
                        ax2.plot(x_ld_list[i], y_ld_list[i], color=colors[i], lw=4, label='LDT')
                        ax2.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None')
                    else:
                        ax1.plot(x_rm_list[i], y_rm_list[i], color=colors[i], lw=4, linestyle='dashed')
                        ax1.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None')
                        ax2.plot(x_ld_list[i], y_ld_list[i], color=colors[i], lw=4)
                        ax2.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None')
            else: 
                for i in range(1, num_graphs, 1):
                    if colors[i] == colors[-1]:
                        ax1.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None')
                        ax1.plot(bin_center_list[i], rho_list[i], color=colors[i], linestyle='None')
                        ax1.plot(x_rm_list[i], y_rm_list[i], color=colors[i], lw=4, label='CLT', linestyle='dashed')

                        ax2.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None')
                        ax2.plot(bin_center_list[i], rho_list[i], color=colors[i], linestyle='None')
                        ax2.plot(x_ld_list[i], y_ld_list[i], color=colors[i], lw=4, label='LDT')
                    else:
                        ax1.plot(x_rm_list[i], y_rm_list[i], color=colors[i], lw=4, linestyle='dashed')
                        ax1.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None')
                        ax2.plot(x_ld_list[i], y_ld_list[i], color=colors[i], lw=4)
                        ax2.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None')
        
        max_y = np.max(y_rm_list[num_graphs-1])
        max_y_index = np.argmax(y_rm_list[num_graphs-1])
        max_x = x_rm_list[num_graphs-1][max_y_index]
        max_x = np.zeros(num_graphs)
        max_y = np.zeros(num_graphs)
        for i in range(num_graphs):
            max_y[i] = np.max(y_rm_list[i])
            max_y_index = np.argmax(y_rm_list[i])
            max_x[i] = x_rm_list[i][max_y_index]

        

        ax1.set_ylabel(r'$p(s)$', fontsize=ylabel_fontsize_scale * fig_width_in, labelpad=y_labelpad_scale * fig_width_in)
        ax2.set_ylabel(r'$p(s)$', fontsize=ylabel_fontsize_scale * fig_width_in, labelpad=y_labelpad_scale * fig_width_in)
        ax2.set_xlabel(r'$s$', fontsize=xlabel_fontsize_scale * fig_width_in, labelpad = x_labelpad_scale * fig_width_in)
        ax1.tick_params(labelsize=tick_labelsize_scale * fig_width_in, pad=tick_labelpad_scale * fig_width_in)
        ax2.tick_params(labelsize=tick_labelsize_scale * fig_width_in, pad=tick_labelpad_scale * fig_width_in)
        ax1.tick_params(which='major', length=8, width=2, direction='in')
        ax2.tick_params(which='major', length=8, width=2, direction='in')
        line_width = 3
        ax1.spines['top'].set_linewidth(line_width)
        ax1.spines['bottom'].set_linewidth(line_width)
        ax1.spines['left'].set_linewidth(line_width)
        ax1.spines['right'].set_linewidth(line_width)
        ax2.spines['top'].set_linewidth(line_width)
        ax2.spines['bottom'].set_linewidth(line_width)
        ax2.spines['left'].set_linewidth(line_width)
        ax2.spines['right'].set_linewidth(line_width)



        ########### SPECIFIC PARAMETERS FOR EACH PLOT ################
        """
        # 2D rho=0.285
        ax1.set_ylim(0,5)
        ax2.set_ylim(0,5)  
        ax2.set_xlim(0,1)
        ax1.set_yticks([0, 2, 4]) 
        ax2.set_yticks([0, 2, 4]) 
        ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
        ax2.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])  
        ax1.text(0.48, 4.4, r'2D, $\rho^*=0.285$', fontsize=30)
        ax2.text(0.48, 4.4, r'2D, $\rho^*=0.285$', fontsize=30)
        ax1.text(0.59, 3.8, '\u26AB\u26AB\u26AB', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax2.text(0.59, 3.8, '\u26AB\u26AB\u26AB', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax1.text(0.73, 3.8, 'Simulation', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax2.text(0.73, 3.8, 'Simulation', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax1.legend(loc='upper right', bbox_to_anchor=(0.88, 0.75), prop={'size': legend_fontsize_scale * fig_width_in}, frameon=False)
        ax2.legend(loc='upper right', bbox_to_anchor=(0.88, 0.75), prop={'size': legend_fontsize_scale * fig_width_in}, frameon=False)
        ax1.text(0.045, 0.9, '(a)', transform=ax1.transAxes, fontsize=30, weight='bold')
        ax2.text(0.045, 0.9, '(b)', transform=ax2.transAxes, fontsize=30, weight='bold')
        ax1.text(max_x[3] + 0.1, max_y[3] - 0.5, fr'N={self.N[3]}', fontsize = fig_width_in * text_scale, color=colors[3])
        ax1.text(max_x[2] + 0.15, max_y[2] - 0.2, fr'{self.N[2]}', fontsize = fig_width_in * text_scale, color=colors[2])
        ax1.text(max_x[1] + 0.25, max_y[1] - 0.2, fr'{self.N[1]}', fontsize = fig_width_in * text_scale, color=colors[1])
        #ax1.text(max_x[0] + 0.075, max_y[0] - 1.5, fr'{self.N[0]}', fontsize = fig_width_in * text_scale, color=colors[0])
        ax2.text(max_x[3] + 0.1, max_y[3] - 0.5, fr'N={self.N[3]}', fontsize = fig_width_in * text_scale, color=colors[3])
        ax2.text(max_x[2] + 0.15, max_y[2] - 0.2, fr'{self.N[2]}', fontsize = fig_width_in * text_scale, color=colors[2])
        ax2.text(max_x[1] + 0.25, max_y[1] - 0.2, fr'{self.N[1]}', fontsize = fig_width_in * text_scale, color=colors[1])
        #ax2.text(max_x[0] + 0.1, max_y[0] - 1.5, fr'{self.N[0]}', fontsize = fig_width_in * text_scale, color=colors[0])
        """
        """
        # 3D rho=0.300
        ax1.set_ylim(0,20)
        ax2.set_ylim(0,20)  
        ax2.set_xlim(0,0.5)
        ax1.set_yticks([0, 5, 10, 15]) 
        ax2.set_yticks([0, 5, 10, 15]) 
        ax2.set_xticks([0, 0.2, 0.4]) 
        ax2.set_xticklabels([0, 0.2, 0.4])  
        ax1.text(0.25, 18, r'3D, $\rho^*=0.300$', fontsize=30)
        ax2.text(0.25, 18, r'3D, $\rho^*=0.300$', fontsize=30)
        ax1.text(0.045, 0.9, '(a)', transform=ax1.transAxes, fontsize=30, weight='bold')
        ax2.text(0.045, 0.9, '(b)', transform=ax2.transAxes, fontsize=30, weight='bold')
        ax1.text(0.28, 15, '\u26AB\u26AB\u26AB', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax2.text(0.28, 15, '\u26AB\u26AB\u26AB', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax1.text(0.35, 15, 'Simulation', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax2.text(0.35, 15, 'Simulation', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax1.legend(loc='upper right', bbox_to_anchor=(0.84, 0.75), prop={'size': legend_fontsize_scale * fig_width_in}, frameon=False)
        ax2.legend(loc='upper right', bbox_to_anchor=(0.84, 0.75), prop={'size': legend_fontsize_scale * fig_width_in}, frameon=False)
        ax1.text(max_x[3]+0.03 , max_y[3]-2 , fr'N={self.N[3]}', fontsize = fig_width_in * text_scale, color=colors[3])
        ax1.text(max_x[2]+0.04, max_y[2]-2, fr'{self.N[2]}', fontsize = fig_width_in * text_scale, color=colors[2])
        ax1.text(max_x[1]+0.06, max_y[1]-2, fr'{self.N[1]}', fontsize = fig_width_in * text_scale, color=colors[1])
        ax1.text(max_x[0]+0.08, max_y[0]-2, fr'{self.N[0]}', fontsize = fig_width_in * text_scale, color=colors[0])
        ax2.text(max_x[3]+0.03, max_y[3]-2, fr'N={self.N[3]}', fontsize = fig_width_in * text_scale, color=colors[3])
        ax2.text(max_x[2]+0.04, max_y[2] -2, fr'{self.N[2]}', fontsize = fig_width_in * text_scale, color=colors[2])
        ax2.text(max_x[1]+0.06, max_y[1] -2, fr'{self.N[1]}', fontsize = fig_width_in * text_scale, color=colors[1])
        ax2.text(max_x[0]+0.08, max_y[0] -2, fr'{self.N[0]}', fontsize = fig_width_in * text_scale, color=colors[0])
        """

        """
        # 3D rho=0.304
        ax1.set_ylim(0,15)
        ax2.set_ylim(0,15)  
        ax2.set_xlim(0,0.6)
        ax1.set_yticks([0, 4, 8, 12]) 
        ax2.set_yticks([0, 4, 8, 12]) 
        ax2.set_xticks([0, 0.2, 0.4, 0.6]) 
        ax2.set_xticklabels([0, 0.2, 0.4, 0.6])  
        ax1.text(0.3, 13.5, r'3D, $\rho^*=0.304$', fontsize=30)
        ax2.text(0.3, 13.5, r'3D, $\rho^*=0.304$', fontsize=30)
        ax1.text(0.045, 0.9, '(a)', transform=ax1.transAxes, fontsize=30, weight='bold')
        ax2.text(0.045, 0.9, '(b)', transform=ax2.transAxes, fontsize=30, weight='bold')
        ax1.text(0.33, 11.2, '\u26AB\u26AB\u26AB', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax2.text(0.33, 11.2, '\u26AB\u26AB\u26AB', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax1.text(0.42, 11.2, 'Simulation', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax2.text(0.42, 11.2, 'Simulation', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax1.legend(loc='upper right', bbox_to_anchor=(0.84, 0.75), prop={'size': legend_fontsize_scale * fig_width_in}, frameon=False)
        ax2.legend(loc='upper right', bbox_to_anchor=(0.84, 0.75), prop={'size': legend_fontsize_scale * fig_width_in}, frameon=False)
        ax1.text(max_x[3]+0.04 , max_y[3]-1.5 , fr'N={self.N[3]}', fontsize = fig_width_in * text_scale, color=colors[3])
        ax1.text(max_x[2]+0.05, max_y[2]-1.5, fr'{self.N[2]}', fontsize = fig_width_in * text_scale, color=colors[2])
        ax1.text(max_x[1]+0.07, max_y[1]-1.5, fr'{self.N[1]}', fontsize = fig_width_in * text_scale, color=colors[1])
        ax1.text(max_x[0]+0.1, max_y[0]-1.5, fr'{self.N[0]}', fontsize = fig_width_in * text_scale, color=colors[0])
        ax2.text(max_x[3]+0.04, max_y[3]-1.5, fr'N={self.N[3]}', fontsize = fig_width_in * text_scale, color=colors[3])
        ax2.text(max_x[2]+0.05, max_y[2]-1.5, fr'{self.N[2]}', fontsize = fig_width_in * text_scale, color=colors[2])
        ax2.text(max_x[1]+0.07, max_y[1]-1.5, fr'{self.N[1]}', fontsize = fig_width_in * text_scale, color=colors[1])
        ax2.text(max_x[0]+0.1, max_y[0]-1.5, fr'{self.N[0]}', fontsize = fig_width_in * text_scale, color=colors[0])
        """


        plt.savefig(fr'./publication_final_graphs/{self.D}D_rho_0{self.rho}_scaling.jpeg', dpi=600, bbox_inches='tight')


    def finite_scaling_one_plot(self):
        colors=['cyan', 'red', 'blue', 'black']
        labels=[str(self.N1), str(self.N2), str(self.N3), str(self.N4)]
        
        num_graphs = len(colors)
        num_bins = 39

        data_list = []
        x_rm_list = []
        x_ld_list = []
        y_rm_list = []
        y_ld_list = []

        rho_list = []
        bins_list = []
        bin_width_list = []
        bin_center_list = []

        for i in range(num_graphs):
            
            data_array = np.load(f'data/{self.D}D_gay_berne/rho_0{self.rho}/N{self.N[i]}/total_first.npy')
            data_list.append(data_array)

            x_rm_array = np.load(f'results/{self.D}D_gay_berne/rho_0{self.rho}/N{self.N[i]}/central_limit_{self.D}D_x.npy')
            x_rm_list.append(x_rm_array)
            y_rm_array = np.load(f'results/{self.D}D_gay_berne/rho_0{self.rho}/N{self.N[i]}/central_limit_{self.D}D_y.npy')
            y_rm_list.append(y_rm_array)
            
            x_ld_array = np.load(f'results/{self.D}D_gay_berne/rho_0{self.rho}/N{self.N[i]}/large_deviations_{self.D}D_x.npy')
            x_ld_list.append(x_ld_array)
            y_ld_array = np.load(f'results/{self.D}D_gay_berne/rho_0{self.rho}/N{self.N[i]}/large_deviations_{self.D}D_y.npy')
            y_ld_list.append(y_ld_array)
    
        for i in range(num_graphs):
            rho, bins, _ = plt.hist(data_list[i], bins=num_bins, density=True, color=colors[i], label=labels[i])
            rho_list.append(rho)
            bins_list.append(bins)
            bin_width_list.append(bins[1] - bins[0])
            bin_center_list.append((bins[:-1] + bins[1:]) / 2)
            plt.clf() 
     

        fig_width_in = 8.27
        fig_height_in = 11.69
        legend_fontsize_scale = 3.5
        tick_labelsize_scale = 4
        tick_labelpad_scale = 1.2
        xlabel_fontsize_scale = 6.5
        ylabel_fontsize_scale = 6.5
        y_labelpad_scale = +0.5
        x_labelpad_scale = -0.6
        text_scale = 4


        
        _, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
        if self.rho == '':
            for i in range(0, num_graphs, 1):
                if colors[i] == colors[-1]:
                    ax.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None')
                    ax.plot(x_rm_list[i], y_rm_list[i], color=colors[i], lw=4, label='CLT', linestyle = 'dashed')
                    ax.plot(x_ld_list[i], y_ld_list[i], color=colors[i], lw=4, label='LDT')
                else:
                    ax.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None')
                    ax.plot(x_rm_list[i], y_rm_list[i], color=colors[i], lw=4, linestyle = 'dashed')
                    ax.plot(x_ld_list[i], y_ld_list[i], color=colors[i], lw=4)
        else:
            if self.rho != 285:
                for i in range(0, num_graphs, 1):
                    if colors[i] == colors[-1]:
                        ax.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10,  linestyle='None')
                        ax.plot(bin_center_list[i], rho_list[i], color=colors[i],  linestyle='None')
            
                        ax.plot(x_rm_list[i], y_rm_list[i], color=colors[i], lw=4, label='CLT', linestyle = 'dashed')
                        ax.plot(x_ld_list[i], y_ld_list[i], color=colors[i], lw=4, label='LDT')
                    else:
                        ax.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None')
                        ax.plot(x_rm_list[i], y_rm_list[i], color=colors[i], lw=4, linestyle = 'dashed')
                        ax.plot(x_ld_list[i], y_ld_list[i], color=colors[i], lw=4)
            else: 
                for i in range(1, num_graphs, 1):
                    if colors[i] == colors[-1]:
                        ax.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None', label=fr'$\rho \sigma^{self.D}=0.{self.rho}$')
                        ax.plot(x_rm_list[i], y_rm_list[i], color=colors[i], lw=4, label='CLT', linestyle = 'dashed')
                        ax.plot(x_ld_list[i], y_ld_list[i], color=colors[i], lw=4, label='LDT')
                    else:
                        ax.plot(bin_center_list[i], rho_list[i], color=colors[i], marker='o', markersize=10, linestyle='None')
                        ax.plot(x_rm_list[i], y_rm_list[i], color=colors[i], lw=4, linestyle = 'dashed')
                        ax.plot(x_ld_list[i], y_ld_list[i], color=colors[i], lw=4)

        max_x = np.zeros(num_graphs)
        max_y = np.zeros(num_graphs)
        for i in range(num_graphs):
            max_y[i] = np.max(y_rm_list[i])
            max_y_index = np.argmax(y_rm_list[i])
            max_x[i] = x_rm_list[i][max_y_index]     

        ax.set_ylabel(r'$p(s)$', fontsize=ylabel_fontsize_scale * fig_width_in, labelpad=y_labelpad_scale * fig_width_in)
        ax.set_xlabel(r'$s$', fontsize=xlabel_fontsize_scale * fig_width_in, labelpad = x_labelpad_scale * fig_width_in)
        ax.tick_params(labelsize=tick_labelsize_scale * fig_width_in, pad=tick_labelpad_scale * fig_width_in)
        ax.tick_params(which='major', length=4, width=2, direction='in')
        line_width = 3
        ax.spines['top'].set_linewidth(line_width)
        ax.spines['bottom'].set_linewidth(line_width)
        ax.spines['left'].set_linewidth(line_width)
        ax.spines['right'].set_linewidth(line_width)



        ####### Specific parameters for each graph for JCP level quality ############
        
        
        """
        # 2D_rho0_scaling
        ax.set_ylim(0,28)  
        ax.set_xlim(0,0.30)
        ax.set_yticks([0, 10, 20]) 
        ax.set_xticks([0, 0.1, 0.2, 0.3]) 
        ax.set_xticklabels([0, 0.1, 0.2, 0.3])
        ax.text(0.12, 26, '2D, Uncorrelated', fontsize=30)
        ax.text(0.13, 21.5, '\u26AB\u26AB\u26AB', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax.text(0.185, 21.5, 'Simulation', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax.legend(loc='upper right', bbox_to_anchor=(0.8, 0.75), prop={'size': legend_fontsize_scale * fig_width_in}, frameon=False)   
        ax.text(max_x[3] + 0.018, max_y[3] - 1.5, fr'N={self.N[3]}', fontsize = fig_width_in * text_scale, color=colors[3])
        ax.text(max_x[2] + 0.035, max_y[2] - 2, fr'{self.N[2]}', fontsize = fig_width_in * text_scale, color=colors[2])
        ax.text(max_x[1] + 0.045, max_y[1] - 2, fr'{self.N[1]}', fontsize = fig_width_in * text_scale, color=colors[1])
        ax.text(max_x[0] + 0.065, max_y[0] - 2, fr'{self.N[0]}', fontsize = fig_width_in * text_scale, color=colors[0])
        """

        """
        # 3D_rho0_scaling
        ax.set_ylim(0,60)  
        ax.set_xlim(0,0.2)
        ax.set_yticks([0, 20, 40]) 
        ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2]) 
        ax.set_xticklabels([0, 0.05, 0.1, 0.15, 0.2])
        ax.text(0.082, 56, '3D, Uncorrelated', fontsize=30)
        ax.text(0.087, 46, '\u26AB\u26AB\u26AB', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax.text(0.125, 46, 'Simulation', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax.legend(loc='upper right', bbox_to_anchor=(0.8, 0.75), prop={'size': legend_fontsize_scale * fig_width_in}, frameon=False)   
        ax.text(max_x[3] + 0.01, max_y[3] - 2, fr'N={self.N[3]}', fontsize = fig_width_in * text_scale, color=colors[3])
        ax.text(max_x[2] + 0.012, max_y[2] - 3, fr'{self.N[2]}', fontsize = fig_width_in * text_scale, color=colors[2])
        ax.text(max_x[1] + 0.020, max_y[1] - 3, fr'{self.N[1]}', fontsize = fig_width_in * text_scale, color=colors[1])
        ax.text(max_x[0] + 0.025, max_y[0] - 3, fr'{self.N[0]}', fontsize = fig_width_in * text_scale, color=colors[0])
        """
        
        """
        # 2D_rho023_scaling
        ax.set_ylim(0,15)  
        ax.set_xlim(0,0.50)
        ax.set_yticks([0, 4, 8, 12]) 
        ax.set_xticks([0, 0.1, 0.3, 0.5]) 
        ax.set_xticklabels([0, 0.1, 0.3, 0.5])
        ax.text(0.24, 14, r'2D, $\rho^*=0.230$', fontsize=30)
        ax.text(0.215, 11.2, '\u26AB\u26AB\u26AB', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax.text(0.31, 11.2, 'Simulation', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax.legend(loc='upper right', bbox_to_anchor=(0.8, 0.75), prop={'size': legend_fontsize_scale * fig_width_in}, frameon=False)   
        ax.text(max_x[3] + 0.028, max_y[3] - 1.3, fr'N={self.N[3]}', fontsize = fig_width_in * text_scale, color=colors[3])
        ax.text(max_x[2] + 0.050, max_y[2] - 1.8, fr'{self.N[2]}', fontsize = fig_width_in * text_scale, color=colors[2])
        ax.text(max_x[1] + 0.085, max_y[1] - 1.4, fr'{self.N[1]}', fontsize = fig_width_in * text_scale, color=colors[1])
        ax.text(max_x[0] + 0.115, max_y[0] - 1, fr'{self.N[0]}', fontsize = fig_width_in * text_scale, color=colors[0])
        """
        """
        # 3D_rho0_scaling
        ax.set_ylim(0,26)  
        ax.set_xlim(0,0.35)
        ax.set_yticks([0, 10, 20]) 
        ax.set_xticks([0, 0.1, 0.2, 0.3]) 
        ax.set_xticklabels([0, 0.1, 0.2, 0.3])
        ax.text(0.16, 24, r'3D, $\rho^*=0.280$', fontsize=30)
        ax.text(0.16, 19.5, '\u26AB\u26AB\u26AB', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax.text(0.225, 19.5, 'Simulation', color='black', fontsize=legend_fontsize_scale * fig_width_in, va='center')
        ax.legend(loc='upper right', bbox_to_anchor=(0.82, 0.75), prop={'size': legend_fontsize_scale * fig_width_in}, frameon=False)   
        ax.text(max_x[3] + 0.013, max_y[3] - 2, fr'N={self.N[3]}', fontsize = fig_width_in * text_scale, color=colors[3])
        ax.text(max_x[2] + 0.026, max_y[2] - 2.5, fr'{self.N[2]}', fontsize = fig_width_in * text_scale, color=colors[2])
        ax.text(max_x[1] + 0.035, max_y[1] - 2.5, fr'{self.N[1]}', fontsize = fig_width_in * text_scale, color=colors[1])
        ax.text(max_x[0] + 0.054, max_y[0] - 2.5, fr'{self.N[0]}', fontsize = fig_width_in * text_scale, color=colors[0])
        """
        
        ###### SAVE NOT DISPLAY ##########
        plt.savefig(fr'./publication_final_graphs/{self.D}D_rho_0{self.rho}_scaling.jpeg', dpi=600, bbox_inches='tight')