import pandas as pd
import matplotlib.pyplot as plt


dt = pd.read_csv('datasets/dataset_train.csv')
courses_name = dt.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1).columns
