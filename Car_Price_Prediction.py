#!/usr/bin/env python3
# ============================================================================ #
# Car Price Prediction - Frederick T. A. Freeth                     17/08/2023 |
# ============================================================================ #
# Following https://www.youtube.com/watch?v=IA3WxTTPXqQ.

import csv
import tensorflow as tf
import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    # The Task:
    # -----------------
    # We want to predict the price of second-hand cars given several input features.
    # Owners of these cars will specify:
    # - "years": how old the car is,
    # - "km": how many kilometres the car has driven,
    # - "rating": the rating of the vehicle,
    # - "condition": the physical condition of the vehicle,
    # - "economy": the fuel economy of the car,
    # - "top speed": the maximum speed the car can drive,
    # - "hp" the horse-power of the car's engine,
    # - "torque": the engine torque,
    # - "current price": the current price of the car.
    #
    #
    # The Model:
    # -----------------
    # Consider the data of car engine horsepower (hp) and price ($) of the car.
    # Suppose X is the model input, and Y is the model output. We want to then
    # predict two values given an input of car engine horsepower.
    # ___________________________________________________
    # | X(hp) | 109, 144,   113,   97,    ... , 80, 150 |
    # | Y(K$) |   8,   9.3,   7.5,  8.89, ... ,  ?,   ? |
    # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    # We can then have two versions of a model we want to make (Note well the
    # direction of the arrows):
    #
    # 1)  [ X ] ---> [ Model ] <--- [ Y ]
    #
    # 2)  [ X ] ---> [ Model ] ---> [ Y ]
    #
    # In (1), the model is fed both the input and the output. In (2), the model
    # is fed an input, and then an output is produced. Thia first stage is the
    # model training step. Once the model is trained, we can then supply the
    # inputs to get our predicted output.
    #
    #
    # Data Preparation:
    # -----------------
    # Data Source: Mayank Patel, Kaggle.
    # https://www.kaggle.com/datasets/mayankpatel14/second-hand-used-cars-data-set-linear-regression
    # Note: I have removed the "on road old" and the "on road now" features of the dataset.
    #
    # Ignoring vehicle ID ("v.id"), our input X is a tensor of shape (N = 1000, 8)
    # and our output tensor Y is a tensor of shape (N = 1000, 1), since the data
    # has the following structure:
    #
    #         |------------------------------- X -----------------------------| |----- Y -----|
    # __________________________________________________________________________________________
    # | v.id  years  km	     rating  condition  economy  top speed  hp  torque | current price |
    # | ---------------------------------------------------------------------- | ------------- | _
    # |    1   3      78945    1       2           14    177        73  123    | 351318        | |
    # |    2   6     117220    5       9            9    148        74   95    | 285001.5      | |
    # |    3   2     132538    2       8           15    181        53   97    | 215386        | N
    # |  ... ...        ...  ...     ...          ...    ...       ...  ...    |    ...        | |
    # | 1000   5      67295    4       2            8    199        99   96    | 414938.5      | |
    # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾
    # We can now begin with reading in the model data, and converting it to a tensor.

    # Read in the data
    car_data = pd.read_csv("Car_Prices.csv", sep = ",")
    
# ============================================================================ #
# Car Price Prediction - Code End                                              |
# ============================================================================ #
