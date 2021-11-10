from kernels import Kernel
import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_excel('clean_data_mark.xlsx')
df.drop([len(df)-1],axis=0,inplace=True)

observed = df['gyro2'].sample(n=round(len(df)/2))
predicted = df['gyro0'].sample(n=round(len(df)/2))
print(observed)
print(predicted)
k = Kernel(observed, predicted)
data_frame = pd.DataFrame(k.normalize())
data_frame = data_frame.fillna(0)

if __name__ == '__main__':
    print(data_frame)
    data_frame.plot()
    plt.show()