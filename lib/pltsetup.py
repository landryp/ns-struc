#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator

plt.rcParams['axes.color_cycle'] = ['black', 'limegreen', 'darkgrey', 'orange', 'mediumturquoise', 'hotpink', 'blue', 'darkred']
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['figure.figsize'] = 10, 10
AUTO_COLORS =  plt.rcParams['axes.color_cycle']
