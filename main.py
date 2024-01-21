

# MODULES
import numpy as np
from src.modules.data_loader import DataLoader
from src.modules.moment_calculation import MomentCalculation
from src.modules.visualization import Visualization
from src.modules.calculate_trace import CalculateTrace
from src.modules.mean_field import MeanField
from src.modules.analytical_methods import AnalyticalMethods
from src.modules.finite_scaling import FiniteScaling



# CONFIGURATION FILE
import json
import sys


with open(sys.argv[1], 'r') as openfile:
    json_object1 = json.load(openfile)

D = json_object1['D']
rho = json_object1['rho']

num_particles = json_object1['num_particles']
num_directories = json_object1['num_directories']
num_bins = json_object1['num_bins']
base_filename = json_object1['base_filename']
data_path = json_object1['data_path'].replace('{D}', str(D)).replace('{rho}', str(rho)).replace('{num_particles}', str(num_particles))
data_file = json_object1['data_file'].replace('{D}', str(D)).replace('{rho}', str(rho)).replace('{num_particles}', str(num_particles)).replace('{base_filename}', str(base_filename))
results_path = json_object1['results_path'].replace('{D}', str(D)).replace('{rho}', str(rho)).replace('{num_particles}', str(num_particles))
results_file = json_object1['results_file'].replace('{D}', str(D)).replace('{rho}', str(rho)).replace('{num_particles}', str(num_particles)).replace('{base_filename}', str(base_filename))



############ Run my modules, RUNNNNN ##############


#dl = DataLoader(data_path, base_filename, num_directories, num_particles)
#dl.load_txtdata()
#dl.save_as_npy(data_file)



"""
mc = MomentCalculation(data_file)
mean_value = mc.my_mean()
std_value = mc.my_stdeav()
with open(results_file, "w+") as f:
    f.write("{:.10f}".format(mean_value))
    f.write(" ")
    f.write("{:.10f}".format(std_value))
"""



"""
vs = Visualization(num_particles)
vs.timeseries_file(base_filename, data_file)
"""



"""
with open(sys.argv[2], 'r') as openfile:
    json_object2 = json.load(openfile)
lower_limit = json_object2['lower_limit']
upper_limit = json_object2['upper_limit']
steps = json_object2['steps']
with open(f'{results_path}/mean_chi.txt', "r") as file:
    average_chi = float(file.read().split()[0]) 
    mf = MeanField(data_file, num_particles, average_chi, num_bins, 3, 304)
    mf.calculate_betaepsilon(lower_limit, upper_limit, steps)
"""




smallest_num = float('inf')
opt_be = None
with open(f'{results_path}/be.txt', 'r') as file:
    for line in file:
        if line.startswith('be:'):
            line_parts = line.split(',')
            current_be = float(line_parts[0].split(':')[1].strip())
            current_num = float(line_parts[1].split(':')[1].strip())
            if current_num < smallest_num:
                smallest_num = current_num
                opt_be = current_be
with open(f'{results_path}/mean_chi.txt', "r") as file:
    average_chi = float(file.read().split()[0]) 



am = AnalyticalMethods(num_particles, average_chi, opt_be)
s = np.arange(0, 1, 0.001)




clt3d = am.central_limit_3D(s)
#np.save(f'{results_path}/central_limit_3D_y.npy', clt3d)
#np.save(f'{results_path}/central_limit_3D_x.npy', s)
xldt3d, yldt3d, _ = am.large_deviations_3D(s) 
#np.save(f'{results_path}/large_deviations_3D_y.npy', yldt3d)
#np.save(f'{results_path}/large_deviations_3D_x.npy', xldt3d)

clt2d = am.central_limit_2D(s)
#np.save(f'{results_path}/central_limit_2D_y.npy', clt2d)
#np.save(f'{results_path}/central_limit_2D_x.npy', s)
ldt2d = am.large_deviations_2D(s)
#np.save(f'{results_path}/large_deviations_2D_y.npy', ldt2d)
#np.save(f'{results_path}/large_deviations_2D_x.npy', s)



vis = Visualization(num_particles, D, rho)
#vis.multiple_densities(num_bins, data_file,'red', 'Simulation', s, clt3d, 'blue', 'Central Limit Theory', xldt3d, yldt3d, 'black', 'Pade')
vis.multiple_rates(39, data_file,'red', '', s, clt3d, 'blue', 'CLT', xldt3d, yldt3d, 'black', 'LDT')



vis = Visualization(num_particles, D, rho)
#vis.multiple_densities(num_bins, data_file,'red', 'Simulation', s, clt2d, 'blue', 'Central Limit Theory', s, ldt2d, 'black', 'Pade')
#vis.multiple_rates(39, data_file,'red', '', s, clt2d, 'blue', 'CLT', s, ldt2d, 'black', 'LDT')


#N1 = json_object1['N1']
#N2 = json_object1['N2']
#N3 = json_object1['N3']
#N4 = json_object1['N4']
#fs = FiniteScaling(D, rho, N1, N2, N3, N4)
#fs.finite_scaling_one_plot()
#fs.finite_scaling_subplots()