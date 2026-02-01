import eazygrad as ez
import eazygrad.nn as nn
# import torch
# import torch.nn as t_nn
import numpy as np

dataset = ez.data.MNISTDataset()
n_epoch = 1000

# class TorchModel(t_nn.Module):

#     def __init__(self, in_dim, out_dim, h_dim, n_layer=2):
#         self.net = nn.ModuleList()
#         self.net.append(nn.Linear(n_in=in_dim, n_out=h_dim))
#         for _ in range(n_layer-1):
#             self.net.append(nn.Linear(n_in=h_dim, n_out=h_dim))
#         self.net.append(nn.Linear(n_in=h_dim, n_out=out_dim))


#     def forward(self, x):
#         y = x
#         for i in range(len(self.net)-1):
#             y = self.net[i](y)
#             y = ez.relu(y)
#         return self.net[-1](y)

class Model(nn.Module):

    def __init__(self, in_dim, out_dim, h_dim, n_layer=2):
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(n_in=in_dim, n_out=h_dim))
        for _ in range(n_layer-1):
            self.net.append(nn.Linear(n_in=h_dim, n_out=h_dim))
        self.net.append(nn.Linear(n_in=h_dim, n_out=out_dim))


    def forward(self, x):
        y = x
        for i in range(len(self.net)-1):
            y = self.net[i](y)
            y = ez.relu(y)
        return self.net[-1](y)

m = Model(in_dim=784, out_dim=10, h_dim=128)
print(m.net)
optimizer = ez.SGD(m.net.parameters())

for e in range(n_epoch):
    start_idx = np.random.randint(len(dataset.data)-128)
    input_ = ez.from_numpy(dataset.data[start_idx:start_idx+128]).reshape(-1, 28*28)
    targets = ez.from_numpy(dataset.targets[start_idx:start_idx+128])
    print(e)
    optimizer.zero_grad()
    y = m(input_)
    print(np.argmax(y.numpy(), axis=-1)-targets)
    print()
    loss = ez.cross_entropy_loss(y, targets)
    # y_t = torch.from_numpy(y)
    # targets_t = torch.from_numpy(targets)
    # loss_t = torch.cross_entropy_loss(y_t, targets_t)
    # print(loss_t)
    # ez.dag.plot()
    loss.backward()
    optimizer.step()
    print(loss)
    