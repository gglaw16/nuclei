
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fcnn import Net
import pdb
import data as d
import random
import time
import cv2
import math
import os
import numpy as np
from pprint import pprint
from utils import *
import girder as g


#=================================================================================


def train(net, data, params):

    for epoch in range(params['num_epochs']):
        print("==== Epoch %d"%epoch)
        for batch in range(params['num_batches']):
            print("== Batch %d"%batch)
            tile_array, label_array = data.sample_batch(params['minibatch_size'])
            print("--done sampling")
            tile_tensor = torch.from_numpy(tile_array).float()
            #upload_batch_to_girder(tile_tensor, "input tiles")
            tile_tensor = tile_tensor.cuda(params['gpu'])
            tile_variable = Variable(tile_tensor)

            label_tensor = torch.from_numpy(label_array).float()
            label_tensor = label_tensor.cuda(params['gpu'])
            label_variable = Variable(label_tensor)

            # learning rate change with batch size?
            # create your optimizer
            optimizer = optim.SGD(net.parameters(), lr=0.005)

            # loss function
            criterion = nn.MSELoss()
    
            # epoch
            start_loss = -1

            for mini in range(params['num_minibatches']):  # loop over the dataset multiple times
                running_loss = 0.0
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs,activations = net(tile_variable)
                #activations is the input activation to each layer group.

                loss = criterion(outputs, label_variable)
                #loss += local_loss(net, activations)
                loss.backward()
                #loss.backward(retain_graph=True)

                optimizer.step()

                # print statistics
                running_loss = loss.data[0]
                if start_loss < 0 :
                    start_loss = running_loss
                #if i % 2000 == 1999:    # print every 2000 mini-batches
                if mini == 0:
                    print(" %d: loss: %.3f" % (mini + 1, running_loss))
                    print("")
                else:
                    print("\033[1A %d: loss: %.3f" % (mini + 1, running_loss))

            # record the final activations in the heatmaps.
            # This is of questionable value since minbatch_size / num pixels is very small. 
            tmp = outputs.data
            tmp = tmp.cpu()
            tmp = tmp.numpy()
            for i in range(tmp.shape[0]):
                activation = tmp[i,0,0,0]
                data.record_activation(i, activation)
            
            # Save the weights
            filename = params['net_filename']
            print("Saving " + filename)
            torch.save(net.state_dict(), filename)
            

def main():
    params = {}
    params['net_filename'] = "model96.pth"
    params['rf_size'] = 96
    params['rf_stride'] = 4
    params['gpu'] = 0
    params['num_epochs'] = 500
    params['num_batches'] = 30
    params['num_minibatches'] = 30
    params['minibatch_size'] = 150
    params['heatmap_decay'] = 0.75
    
    image_dir = ""
    mask_dir = ""
    heatmap_dir = ""

    data = d.data((params['rf_size'], params['rf_size']))
    data.load_from_directories(image_dir, mask_dir, heatmap_dir)

    # train

    # Create the network.
    net = Net()
    print(net)

    # load the weights
    if os.path.isfile(params['net_filename']):
        net.load_state_dict(torch.load(params['net_filename']))
    net.cuda(params['gpu'])
    
    train(net, data, params)

    

if __name__ == '__main__':

    main()























