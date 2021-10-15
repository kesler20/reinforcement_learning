import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 

mark_data = pd.read_excel(r'bionic_society\clean_data_mark.xlsx')
data = mark_data['emg0']
data = np.cumsum(data)
plt.plot(data)
plt.show()
