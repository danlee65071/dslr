import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dt = pd.read_csv('datasets/dataset_train.csv')
courses_name = dt.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1).columns
courses = dt.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1)
corr_matrix = courses.corr()
houses_names = dt['Hogwarts House'].unique()
indexes = []
for course in courses_name:
    abs_corr_matr = np.absolute(corr_matrix[course]).drop(course)
    if len(abs_corr_matr[np.round(abs_corr_matr, 3) == 1].index):
        if course not in indexes:
            cor_course = abs_corr_matr[np.round(abs_corr_matr, 3) == 1].index[0]
            indexes.extend([cor_course, course])
            for house in houses_names:
                x = dt[dt['Hogwarts House'] == house][course].dropna().to_numpy()
                plt.scatter(courses[course], courses[cor_course], alpha=0.2, label=house)
            plt.legend()
            plt.xlabel(course)
            plt.ylabel(cor_course)
            plt.grid()
            plt.show()
