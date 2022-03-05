import sys
import pandas as pd
import numpy as np
import math


class Describe():
    def __init__(self, dt_name: str):
        self.dt_name = dt_name
        self.dt = pd.read_csv(dt_name)
        self.count = self.dt.shape[0]
        self.columns_name = self.dt.columns
        self.params = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

    def __remove_str_cols(self):
        drop_cols = []
        for i, col in enumerate(self.columns_name):
            if self.dt.dtypes[i] == np.dtype(object):
                drop_cols.append(col)
        self.dt.drop(drop_cols, axis=1, inplace=True)
        self.columns_name = self.dt.columns

    def __print_header(self):
        print('{:>10}'.format(' '), end='')
        for col in self.columns_name:
            print('{:>35}'.format(col), end='')
        print()

    def __count_print(self):
        for col in self.columns_name:
            print('{:>35.6f}'.format(len(self.dt[col].dropna())), end='')

    def __mean_print(self):
        for col in self.columns_name:
            qnty = len(self.dt[col].dropna())
            if qnty == 0:
                print('{:>35s}'.format('NaN'), end='')
                continue
            mean_val = self.dt[col].sum() / qnty
            print('{:>35.6f}'.format(mean_val), end='')
        
    def __std_print(self):
        for col in self.columns_name:
            len_col = len(self.dt[col].dropna())
            if len_col == 0:
                print('{:>35s}'.format('NaN'), end='')
                continue
            mean_val = self.dt[col].sum() / len_col
            sigma = np.sqrt(((self.dt[col] - mean_val) ** 2).sum() / (len_col - 1))
            print('{:>35.6f}'.format(sigma), end='')

    def __min_print(self):
        for col in self.columns_name:
            len_col = len(self.dt[col].dropna())
            if len_col == 0:
                print('{:>35s}'.format('NaN'), end='')
                continue
            min_val = self.dt[col][0]
            for i in range(len_col):
                if min_val > self.dt[col][i]:
                    min_val = self.dt[col][i]
            print('{:>35.6f}'.format(min_val), end='')

    def __first_percentile_print(self):
        for col in self.columns_name:
            len_col = len(self.dt[col].dropna())
            if len_col == 0:
                print('{:>35s}'.format('NaN'), end='')
                continue
            sorted_dt = self.dt[col].sort_values()
            index = (len_col + 1) * 0.25 - 1
            if index % 2 != 0:
                f_percentile = (sorted_dt[math.ceil(index)] + sorted_dt[math.floor(index)]) / 2
            else:
                f_percentile = sorted_dt[index]
            print('{:>35.6f}'.format(f_percentile), end='')

    def print_describe(self):
        self.__remove_str_cols()
        self.__print_header()
        for param in self.params:
            print('{:<10}'.format(param), end='')
            if param == 'Count':
                self.__count_print()
            elif param == 'Mean':
                self.__mean_print()
            elif param == 'Std':
                self.__std_print()
            elif param == 'Min':
                self.__min_print()
            elif param == '25%':
                self.__first_percentile_print()
            print()


dt_name = sys.argv[1]
d = Describe(dt_name)
d.print_describe()