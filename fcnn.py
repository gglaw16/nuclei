from __future__ import print_function
import pdb
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        #self.tmp = []
        #layers = self.tmp
        layers = []
        
        components = 3
        out_components = 32
        # (0,1,2) -> (15,16,17)
        for i in range(6):
            layers.append(nn.Conv2d(components, out_components, 3))
            layers.append(nn.BatchNorm1d(out_components, affine=False))
            layers.append(nn.LeakyReLU())
            components = out_components
        # (18)
        layers.append(nn.MaxPool2d(2))
            
        out_components = 64
        # (19,20,21) (22,23,24) (25,26,27) (28,29,30)
        for i in range(4):
            layers.append(nn.Conv2d(components, out_components, 3))
            layers.append(nn.BatchNorm1d(out_components, affine=False))
            layers.append(nn.LeakyReLU())
            components = out_components
        # (31)
        layers.append(nn.MaxPool2d(2))
            
        out_components = 128
        # (32,33,34)(35,36,37)(38,39,40)(41,42,43)(44,45,46)(47,48,49)(50,51,52)(53,54,55)
        for i in range(8):
            layers.append(nn.Conv2d(components, out_components, 3))
            layers.append(nn.BatchNorm1d(out_components, affine=False))
            layers.append(nn.LeakyReLU())
            components = out_components
            
        # (56,57,58)
        layers.append(nn.Conv2d(components, components, 1))
        layers.append(nn.BatchNorm1d(components, affine=False))
        layers.append(nn.LeakyReLU())
        # (59,60,61)
        layers.append(nn.Conv2d(components, 2, 1))
        # bug, this is not supposed to ber here. layers.append(nn.BatchNorm1d(2, affine=False))
        layers.append(nn.LeakyReLU())
        # (62)
        layers.append(nn.Softmax(1))

        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        # Convolutional group 1
        #x = self.layers(x)
        debug = []
        for i in range(len(self.layers)):
            layer = self.layers[i]
            #if type(layer).__name__ == 'LeakyReLU':
            debug.append(x)
            x = layer(x)
        return x, debug
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class EmptyNet(nn.Module):

    def __init__(self):
        super(EmptyNet, self).__init__()

    def copy(self, layers):
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return x, debug
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        

if __name__ == "__main__":
    pdb.set_trace()
    # Create the network.
    bug = Net()

    # load the weights
    filename='model.pth'
    if os.path.isfile(filename):
        bug.load_state_dict(torch.load(filename))

    layers = bug.tmp
    # remove the problem layer (60)
    layers.remove(layers[60])
    
    # Create the network.
    net = EmptyNet()
    net.copy(layers)

    print(net)
    torch.save(net.state_dict(), filename)


    
    """
    net = Net()
    print(net)

    params = list(net.parameters())
    print(len(params))
    for i in range(len(params)):
        print(params[i].size())

    input = Variable(torch.randn(1, 3, 96, 96))
    print(input.size())
    out,d = net(input)
    print(out.size())
    print(out)

    pdb.set_trace()
    # Save the weights
    filename='checkpoint.pth.tar'
    torch.save(net.state_dict(), filename)

    # load the weights
    net.load_state_dict(torch.load(filename))
    """



"""
# Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))


# Mini batches
# For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
# If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.


# loss function
output = net(input)
target = Variable(torch.arange(1, 11))  # a dummy target, for example
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)




# Gradient of loss
net.zero_grad()     # zeroes the gradient buffers of all parameters
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)




# gradient descent
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update






# epoch
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
"""
