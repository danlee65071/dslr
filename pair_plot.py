import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dt = pd.read_csv('datasets/dataset_train.csv').drop('Index', axis=1)
sns.pairplot(dt, hue='Hogwarts House', diag_kind="hist")
plt.show()
