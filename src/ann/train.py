# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:29:36 2022

@author: srhin
"""

# Note...not all of these are being used..

import torch
from torch.nn.functional import normalize
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datetime import datetime
import sklearn
import sklearn.metrics
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------plotting functions (skip to the fun stuff below)
def unscaled_plot():
    x = 0
    if plot_unscaled_data == 1:  # if plotting selected plot
        for y in input_variables:
            test = dayCENT_inputs[:, x]
            _ = plt.hist(test, color='green')
            plot_title = (y, 'Input Unscaled Histogram')
            plt.title(plot_title)
            plt.ylabel("Count")
            plt.show()
            x += 1

    x = 0  # reset generic counter
    if plot_unscaled_data == 1:
        # if plotting selected plot, plot all
        for z in output_variables:
            test = dayCENT_outputs[:, x]
            _ = plt.hist(test, bins=20, color='blue')
            plot_title = (z, 'Output Unscaled Histogram')  # seems silly but wasn't joining text for title.
            plt.title(plot_title)
            plt.ylabel("Count")
            plt.show()
            x += 1


def scaled_plot():
    x = 0  # reset generic counter
    if plot_scaled_data == 1:
        if plot_unscaled_data == 1:  # if plotting selected plot
            for y in input_variables:
                test = dayCENT_inputs[:, x]
                _ = plt.hist(test, bins=20, color='orange')
                plot_title = (y, 'Input Scaled Histogram')
                plt.title(plot_title)
                plt.ylabel("Count")
                plt.show()
                x += 1

    x = 0  # reset generic counter
    if plot_scaled_data == 1:
        # if plotting selected plot, plot all
        for z in output_variables:
            test = dayCENT_outputs[:, x]
            _ = plt.hist(test, bins=20)
            plot_title = (z, 'Output Scaled Histogram')  # seems silly but wasn't joining text for title.
            plt.title(plot_title)
            plt.ylabel("Count")
            plt.show()
            x += 1


def plot_losses():
    plt.plot(losses, color='blue', label='model')
    plt.plot(val_losses, color='red', label='validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    text_1 = "Learning rate = %f" % (learning_rate)
    text_2 = "/Batch = %i" % (batch)
    text_3 = "/Dropout rate = %f" % (dropout_rate)
    concat_title = text_1 + text_2 + text_3
    plt.title(concat_title)
    plt.legend(loc='best')
    plt.show()
    print("start loss,", max(losses))
    print("end loss,", min(losses))


def plot_model_test_outputs():  # compares model output to test data outputs.
    x = 0  # reset counter variable
    for z in output_variables:
        test = dayCENT_outputs_test[:, x]
        mod = test_output_npy[:, x]
        plt.style.use('seaborn-deep')
        _ = plt.hist([test, mod], bins=20, label=['test data', 'model'])
        plot_title = (z, 'Model Test Run Output, unscaled Histogram')  # seems silly but wasn't joining text for title.
        plt.title(plot_title)
        plt.legend(loc='upper right')
        plt.ylabel("Count")
        plt.show()
        x += 1


def plot_model_test_scatter():
    x = 0
    for z in output_variables:
        _ = plt.scatter(dayCENT_outputs_test[:, x], test_output_npy[:, x])
        plot_title = (z)
        plt.title(plot_title)
        plt.ylabel("Test Model Output")
        plt.xlabel("DayCENT")
        plt.axis('equal')
        plt.show()
        x += 1


"""
def plot_model_test_outputs():
    x=0 #reset counter variable
    for z in output_variables:
        test = dayCENT_outputs[:,x]
        _ = plt.hist(test, bins = 20, color = 'purple')
        plot_title = (z,'Model Test Run Output, unscaled Histogram') #seems silly but wasn't joining text for title. 
        plt.title(plot_title)
        plt.ylabel("Count")
        plt.show()
        x+=1
"""
# **************************************************************************
# **************************************************************************
# **********************Model input/configure/train below*******************
# **************************************************************************
# **************************************************************************

# input training data
# input order  [stover, fertilization, irrigation]:
dayCENT_inputs = np.load('corn_10000_in_train_tensor.npy')
dayCENT_outputs = np.load('corn_10000_out_train_tensor.npy')
size_inputs = dayCENT_inputs.shape
n_input = size_inputs[1]  # calculate the input node count for model
size_outputs = dayCENT_outputs.shape
n_out = size_outputs[1]  # calculate output node count for model

input_variables = ['astgc', 'omad_day', 'clteff_1', 'clteff_2', 'clteff_3', 'clteff_4', 'cult_nt_date', 'cult_b_date',
                   'feramt_n7_1', 'feramt_n7_2', 'feramt_n7_3', 'fert_n7_date', 'crop_c6_date', 'pltm_date',
                   'feramt_n10_1', 'feramt_n10_2', 'feramt_n10_3', 'feramt_n10_date', 'rmvstr', 'remwsd', 'harv_g_date']
output_variables = ['somtc', 'somsc', 'agcprd', 'cgrain', 'stemp']

################# define learning model params
n_hidden = 30  # number of hidden layers
batch, learning_rate, epochrange = 8000, .001, 10000
momentum_SGD = 2.0
weight_decay_SGD = .2
dampening_SGD = 0.0
maximize_SGD = False
bias_y_n = True
dropout_rate = 0.0
torch.manual_seed(42)
################# define activation function
activation_function = nn.Tanh()
# activation_function=nn.ReLU()


# define scalar

scaler_x = MinMaxScaler([-1, 1])
scaler_y = MinMaxScaler([-1, 1])
# scaler_x = MinMaxScaler([0,1])
# scaler_y = MinMaxScaler([0,1])
# scaler_x = PowerTransformer(method="yeo-johnson")
# scaler_y = PowerTransformer(method="yeo-johnson")
# scaler_x = RobustScaler(quantile_range=(10,90))
# scaler_y = RobustScaler(quantile_range=(10,90))


# input test data
dayCENT_inputs_test = np.load('corn_10000_in_test_tensor.npy')
dayCENT_outputs_test = np.load('corn_10000_out_test_tensor.npy')
print("check tensor sizes for APE/MSE/RMSE Calcs", dayCENT_inputs.size, dayCENT_outputs.size, dayCENT_inputs_test.size,
      dayCENT_outputs_test.size)

# define plots to display
plot_unscaled_data = 0  # 1 = plot/0=don't plot
plot_scaled_data = 0

unscaled_plot()  # plot input variables unscaled

# scale inputs/outputs

dayCENT_inputs = scaler_x.fit_transform(dayCENT_inputs)
dayCENT_inputs_test = scaler_x.transform(dayCENT_inputs_test)
dayCENT_outputs = scaler_y.fit_transform(dayCENT_outputs)
dayCENT_outputs_test = scaler_y.transform(dayCENT_outputs_test)

#################  DayCENT Inputs/Outputs Distribution plots
##Inputs

scaled_plot()  # plot input variables scaled

############### convert inputs/outputs to Tensors

x_tensor = torch.from_numpy(dayCENT_inputs)
y_tensor = torch.from_numpy(dayCENT_outputs)
x_test_tensor = torch.from_numpy(dayCENT_inputs_test)
y_test_tensor = torch.from_numpy(dayCENT_outputs_test)

train_data = TensorDataset(x_tensor, y_tensor)

train_loader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, num_workers=0)

############## model
model = nn.Sequential(nn.Linear(n_input, n_hidden, bias_y_n),
                      nn.Dropout(p=dropout_rate, inplace=True),  ### New dropout layer added
                      activation_function,

                      # uncomment for 2 layer model
                      # nn.Linear(n_hidden, n_hidden, bias_y_n),
                      # nn.Dropout(p=dropout_rate,inplace=True),
                      # activation_function,

                      # uncomment for 3 layer model
                      # nn.Linear(n_hidden, n_hidden, bias_y_n),
                      # nn.Dropout(p=dropout_rate,inplace=True),
                      # activation_function,

                      nn.Linear(n_hidden, n_out, bias_y_n))

print(model)
model.zero_grad()  # Zero gradiants, not sure this is necessary here....
model.train()
loss_function = nn.MSELoss()
# loss_function = nn.L1Loss()
# loss_function = nn.NLLLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, maximize = maximize_SGD)

def make_train_step(model, loss_function, optimizer):
    # builds function that performs a step in the train loop
    def train_step(x, y):
        model.train()
        pred_y = model(x_tensor)
        loss = loss_function(pred_y, y_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return train_step


train_step = make_train_step(model, loss_function, optimizer)

losses = []
val_losses = []
batch_count = 1
epoch_count = 0
for epoch in range(epochrange):
    batch_count = 1
    epoch_count += 1
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        loss = train_step(x_batch, y_batch)
        # print("Batch = ",batch_count,"Epoch = ",epoch_count,"Loss = ", loss)
        batch_count += 1
        if batch_count == (
                (x_tensor.size(dim=0) / batch) + 1):  # only give the loss at the last batch, decimating loss.
            print("Batch = ", batch_count, "Epoch = ", epoch_count, "Loss = ", loss)
            losses.append(loss)

    # model.train()
    # pred_y = model(x_tensor)
    # loss = loss_function(pred_y, y_tensor)
    # losses.append(loss.item())
    # model.zero_grad()
    # loss.backward()
    # optimizer.step()  #old model traing code.
    with torch.no_grad():
        model.eval()
        yhat = model(x_test_tensor)
        val_loss = loss_function(yhat, y_test_tensor)
        val_losses.append(val_loss.item())

plot_losses()

model.eval()  # set model into eval mode
model_output = model(x_test_tensor)  # run model.
test_output_npy = model_output.detach().numpy()  # convert from tensor back to numpy
test_output_npy = scaler_y.inverse_transform(test_output_npy)  # perform inverse transform.
dayCENT_outputs_test = scaler_y.inverse_transform(dayCENT_outputs_test)

plot_model_test_outputs()

x = 0  # reset counter variable
for z in output_variables:
    parameter_ape = mean_absolute_error(dayCENT_outputs_test[:, x], test_output_npy[:, x])
    print(z, '_MAE', parameter_ape)
    parameter_mape = mean_absolute_percentage_error(dayCENT_outputs_test[:, x], test_output_npy[:, x])
    print(z, '_MAPE', parameter_mape)
    x += 1

plot_model_test_scatter()

"""
x=0 #reset counter variable
for z in output_variables:
    test = dayCENT_outputs_test[:,x]
    mod = test_output_npy[:,x]
    plt.style.use('seaborn-deep')
    _ = plt.hist([test,mod], bins = 20,label=['test data','model'])
    plot_title = (z,'Model Test Run Output, unscaled Histogram') #seems silly but wasn't joining text for title. 
    plt.title(plot_title)
    plt.ylabel("Count")
    plt.show()
    x+=1
        
"""
