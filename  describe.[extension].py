import sys
import pandas as pd
import math

class Describe():
    def __init__(self, dt_name: str):
        self.dt_name = dt_name
        self.dt = pd.read_csv(dt_name)
        self.count = self.dt.shape[0]
        self.columns_name = self.dt.columns
        self.params = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

    def __print_header(self):
        print('{:>10}'.format(' '), end='')
        for col in self.columns_name:
            print('{:>20}'.format(col), end='')
        print()
        
    def __std_print(self):
        for col in self.columns_name:
            mean_val = self.dt[col].sum() / self.count
            sigma = 0.0
            for i in range(self.count):
                sigma += (self.dt[col][i] - mean_val) ** 2
            sigma = math.sqrt(sigma / (self.count - 1))
            print('{:>20.6f}'.format(sigma), end='')

    def print_describe(self):
        self.__print_header()
        for param in self.params:
            print('{:<10}'.format(param), end='')
            if param == 'Count':
                for col in self.columns_name:
                    print('{:>20.6f}'.format(self.count), end='')
            elif param == 'Mean':
                for col in self.columns_name:
                    mean_val = self.dt[col].sum() / self.count
                    print('{:>20.6f}'.format(mean_val), end='')
            elif param == 'Std':
                self.__std_print()
            print()



dt_name = sys.argv[1]
d = Describe(dt_name)
d.print_describe()
