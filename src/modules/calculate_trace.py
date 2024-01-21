
class CalculateTrace:
    def __init__(self, npy_file1, npy_file2):
        self.s1 = np.load(npy_file1)
        self.s2 = np.load(npy_file2)
        self.s3 = - (self.s1 + self.s2)

    def trace_squared(self):
        square = self.s1**2 + self.s2**2 + self.s3**2
        np.save("trace_squared.npy", square)
     
    def trace_cubed(self):
        cube = self.s1**3 + self.s2**3 + self.s3**3
        np.save("trace_cubed.npy", cube)