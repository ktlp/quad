import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.utils.data as data
import matplotlib.pyplot as plt
import imageio
import torch

# sample generator
# f(x) = ax^2 + bx + c
# min @ -b/2a
tt = 2
def gen(vector):
    assert isinstance(vector,np.ndarray), 'Not correct type'
    return 2*np.power(vector,2) + 2*vector + 1

class pol():
    def __init__(self,a,b,c,vector):
        self.vector = vector
        self.min = -b/(2*a)
        self.data = a*np.power(vector,2) + b*vector + c
        return

    def __call__(self):
        return self.data

# randomly create 2K pol instances, with same vectors
x = np.linspace(-100,200,2000)
min,x1,x2,x3,y1,y2,y3 = [],[],[],[],[],[],[]
for i in range(2000):
    inds = np.random.randint(len(x),size = 3)
    vec = x[inds]
    a,b,c = tuple(np.random.randn(3))
    temp_pol = pol(a,b,c,x)
    min.append(temp_pol.min)
    x1.append(vec[0])
    x2.append(vec[1])
    x3.append(vec[2])
    y1.append(temp_pol()[0])
    y2.append(temp_pol()[1])
    y3.append(temp_pol()[2])

df = pd.DataFrame({'min':min,'x1':x1,'x2':x2,'x3':x3,'y1':y1,'y2':y2,'y3':y3})
temp2 = 1
class PandasDataSet(TensorDataset):
    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet,self).__init__(*tensors)

    def _df_to_tensor(self,df):
        if isinstance(df, pd.Series):
            df = df.to_frame()
        return torch.from_numpy(df.values).float()
df_y = df['min'].copy()
df_x = df.drop(labels=['min'],axis=1)
train_set = PandasDataSet(df_x,df_y)
loader = DataLoader(train_set,batch_size=32,shuffle=True,drop_last=False)

class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden_0, n_hidden_1, n_output):
        super(Net, self).__init__()
        self.hidden_0 = torch.nn.Linear(in_features=n_features, out_features=n_hidden_0)
        self.hidden_1 = torch.nn.Linear(in_features=n_hidden_0, out_features=n_hidden_1)
        self.predict = torch.nn.Linear(n_hidden_1,n_output)

    def forward(self, x):
        x = F.leaky_relu(self.hidden_0(x))
        x = F.leaky_relu(self.hidden_1(x))
        x = self.predict(x)
        return x

net = Net(6,50,50,1)
optimizer = torch.optim.SGD(net.parameters(), lr = 0.2)
loss_func = torch.nn.MSELoss()
n_epochs = 1000

# training
for epoch in range(n_epochs):
    print('Progress: {} %'.format((epoch+1)/n_epochs*100))
    for x,y in loader:
        optimizer.zero_grad()
        prediction = net(x)
        loss = loss_func(prediction,y)
        loss.backward()
        optimizer.step()

df = pd.DataFrame({'x1':np.array([0,2]),'x2':np.array([1,3]),'x3':np.array([0.5,9]),'y1':np.array([1,1]),'y2':np.array([4,7]),'y3':np.array([2.25,95])})
print(df.head())
train_set = PandasDataSet(df)
loader = DataLoader(train_set,batch_size=1,shuffle=True,drop_last=False)
for x in loader:
    print(x[0])
    print(net(x[0]))
