
import os
import numpy as np

class DataLoader:
    def __init__(self, base_dir_path, base_filename, num_directories, num_particles):
        self.base_dir_path = base_dir_path
        self.base_filename = base_filename
        self.num_directories = num_directories
        self.num_particles = num_particles

    def load_txtdata(self):
        Y = np.array([])

        for i in range(1, self.num_directories + 1):
            dir_path = os.path.join(self.base_dir_path, f"N{self.num_particles}s{i}")
            file_name = f'{self.base_filename}{i}.txt'
            filepath = os.path.join(dir_path, file_name)

            with open(filepath, 'r') as file:
                y_i = np.loadtxt(file, usecols=(0), unpack=True)
                Y = np.concatenate((Y, y_i))

        return Y

    def save_as_npy(self, output_file):
        Y = self.load_txtdata()
        np.save(output_file, Y)