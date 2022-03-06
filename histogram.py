import pandas as pd
import matplotlib.pyplot as plt


dt = pd.read_csv('datasets/dataset_train.csv')
houses_names = dt['Hogwarts House'].unique()
courses_names = dt.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1).columns
for course in courses_names:
    for house in houses_names:
        x = dt[dt['Hogwarts House'] == house][course]
        plt.hist(x, bins=20, label=house, stacked=True, alpha=0.2)
    plt.legend()
    plt.title(course)
    plt.xlabel('Mark')
    plt.ylabel('Student count')
    plt.show()
