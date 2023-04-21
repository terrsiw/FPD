# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from mixture_ratio import MixtureRatio
import numpy as np
from mixture_ratio import MixtureRatio



X = np.array([[0, 0], [1, 0], [1, 1], [1, 1], [2, 1], [3, 2]])
Y = np.array([0, 0, 0, 1, 1, 1])
from mixture_ratio import MixtureRatio
mix = MixtureRatio(variables_domain=[2, 4, 3])
mix.fit(X, Y)
print(mix.predict_proba([[0, 1]]))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
