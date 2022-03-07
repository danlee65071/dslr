import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps


def print_graph(mean_v, std_v):
    if mean_v == 0 and std_v == 1:
        plt.title(course + ' (standard normal distribution)')
    else:
        plt.title(course)
    plt.legend()
    plt.xlabel('Mark')
    plt.ylabel('Student count')
    plt.grid()
    plt.show()


def print_stats(data, course_name, mean_v, std_v):
    print(course_name)
    print('{:<20s}'.format('statistic: '), '{:<8.3f}'.format(sps.anderson(data.dropna())[0]))
    print('{:<20s}'.format('critical value 5%: '), '{:<8.3f}'.format(sps.anderson(data.dropna())[1][2]))
    print('{:<20s}'.format('mean: '), '{:<8}'.format(mean_v))
    print('{:<20s}'.format('std: '), '{:<8}'.format(std_v))


dt = pd.read_csv('datasets/dataset_train.csv')
houses_names = dt['Hogwarts House'].unique()
courses_names = dt.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1).columns
for course in courses_names:
    if sps.anderson(dt[course].dropna())[0] >= sps.anderson(dt[course].dropna())[1][2]:
        continue

    x_norm = np.linspace(dt[course].max(), dt[course].min(), dt[course].count())
    mean_val = dt[course].mean()
    std_val = dt[course].std()
    pdf = sps.norm.pdf(x_norm, loc=mean_val, scale=std_val)

    round_mean = np.round(mean_val)
    round_std = np.round(std_val)

    print_stats(dt[course], course, round_mean, round_std)

    plt.plot(x_norm, pdf, label='norm dist')
    plt.hist(dt[course], bins=20, stacked=True, alpha=0.2, density=True)
    print_graph(round_mean, round_std)

    for house in houses_names:
        x = dt[dt['Hogwarts House'] == house][course].dropna().to_numpy()
        plt.hist(x, bins=20, label=house, stacked=True, alpha=0.2)
    print_graph(round_mean, round_std)
    print()
