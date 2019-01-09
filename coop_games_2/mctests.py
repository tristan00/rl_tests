import numpy as np
import theano.tensor as tt
import pymc3 as pm

import seaborn as sns
import matplotlib.pyplot as plt


with pm.Model() as model:
    x = pm.Normal('x', mu=0)
    y = pm.Gamma('y', alpha=1, beta=1)
    print(x)
    plus_2 = x + 2
    summed = x + y
    squared = x ** 2
    sined = pm.math.sin(x)