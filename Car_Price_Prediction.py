#!/usr/bin/env python3
# ============================================================================ #
# Car Price Prediction - Frederick T. A. Freeth                     21/08/2023 |
# ============================================================================ #
# Following https://www.youtube.com/watch?v=IA3WxTTPXqQ.

import csv
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import InputLayer, Normalization, Dense
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber

if __name__ == "__main__":
    
    # The Task:
    # --------------------------
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
    # --------------------------
    # Consider the data of car engine horsepower (hp) and price ($) of the car.
    # Suppose X is the model input, and Y is the model output. We want to then
    # predict two values given an input of car engine horsepower.
    # ___________________________________________________
    # | X(hp) | 109, 144,   113,   97,    ... , 80, 150 |
    # | Y($)  |   8,   9.3,   7.5,  8.89, ... ,  ?,   ? |
    # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    # We can then have two versions of a model we want to make (Note well the
    # direction of the arrows):
    #
    # 1)  [ X ] ---> [ Model ] <--- [ Y ]
    #
    # 2)  [ X ] ---> [ Model ] ---> [ Y ]
    #
    # In (1), the model is fed both the input and the output. In (2), the model
    # is fed an input, and then an output is produced. This first stage is the
    # model training step. Once the model is trained, we can then supply the
    # inputs to get our predicted output.
    #
    #
    # Data Preparation:
    # --------------------------
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
    # |    1    3     78945    1       2           14    177        73  123    | 351318        | |
    # |    2    6    117220    5       9            9    148        74   95    | 285001.5      | |
    # |    3    2    132538    2       8           15    181        53   97    | 215386        | N
    # |  ...  ...       ...  ...     ...          ...    ...       ...  ...    |    ...        | |
    # | 1000    5     67295    4       2            8    199        99   96    | 414938.5      | |
    # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾
    # We can now begin with reading in the model data, and converting it to a tensor.

    # Read in the data
    car_data = pd.read_csv("Car_Prices.csv", sep = ",")

    # Basic data investigation:
    pairplot = pd.plotting.scatter_matrix(
        car_data, figsize = (5, 5), marker = 'o',
        hist_kwds = {'bins': 1000}, s = 120, alpha = 0.4
    )
    plt.show()
    # We can see broadly there aren't many patturns we can see. However, we can
    # observe that as car distance travelled (in km) increases, so does the value
    # of the car.

    # We now convert the data to a tensor of float32 types:
    car_data_tensor = tf.cast(x = tf.constant(car_data), dtype = "float32")

    # We also shuffle the rows of the data to remove any potential underlying
    # biases arising from how the data was collected:
    car_data_tensor = tf.random.shuffle(car_data_tensor)

    # For our model above, we extract our data X and Y. For Y, we use the
    # tf.expand_dims method to turn it into a column tensor
    X = car_data_tensor[:, 0:8]
    y = tf.expand_dims(input = car_data_tensor[:, 9], axis = 1)

    # To help the data train faster, we can normalise the inputs, X. We normalise
    # by subtracting from the mean and dividing by the variance which is the
    # square of the standard deviation: X ^ {tilde} = (X - μ) / (σ ^ 2):
    normaliser = Normalization() # Init normaliser
    normaliser.adapt(X) # Find the mean and standard deviation of each column
    X_normalised = normaliser(X) # Normalise the data

    # Essentially, for each feature from X, say x (lower case x), we can compare
    # it to y via a plot y = mx + c assuming a linear regression. This equation
    # allows us to extrapolate and predict values of y in x for which we have
    # no data on. Here, m is the gradient of out line, but in the context of
    # machine learning, this is called the weights. The value c is the y-intercept
    # and this is called the bias. In summary, y = m X + c is the model. We are
    # trying to train the model such that the values of m and c accurately fit
    # the data (i.e. that best represents the data). To build the machine learning
    # model, we use the Keras Sequential API. We want our model to have a
    # normalisation layer and a dense layer:
    #
    #  Inputs            Model Layers            Outputs
    #  ‾‾‾‾‾‾    [       ‾‾‾‾‾‾‾‾‾‾‾‾       ]    ‾‾‾‾‾‾‾
    #  [   ]     [ [       ]      [       ] ]     [   ]
    #  [ X ] --> [ [ Norm. ] ---> [ Dense ] ] --> [ y ]
    #  [   ]     [ [       ]      [       ] ]     [   ]
    #            [                          ]
    #
    # Our inputs need to be normalised before being used in the dense layer, which
    # is why we need the normalisation layer. The dense layer is a fully connected
    # layer, meaning that each neuron from the previous layer is connected to a
    # neuron in the dense layer forming a bijection between the two layers. What
    # it does is that that it takes an input x, multiply it by a weight, and then
    # add a bias. So, mx + c = y_p which is our predicted value of y. For M = 8
    # features, we connect 8 neurons from the input layer into the normalisation
    # layer which has the same shape as the input so we can normalise each value.
    # From here, these normalised inputs pass into the dense layer which has an
    # output shape of 1, since we want to know what a predicted car's price is.
    # This predicted price is a singular number, which is why the dense layer
    # must have an output of shape of 1.
    #
    # Between the normalisation and dense layers, each normalised input neuron
    # is multiplied by the weight and summed up and added with a weight which is
    # the value of the Dense(1) layer:
    #
    #  x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8
    #   ○   ○   ○   ○   ○   ○   ○   ○   Normalisation Layer
    #    \   \   \   \ /   /   /   /
    #                 ○                 Dense(1) Layer
    #  m_1 m_2 m_3 m_4 m_5 m_6 m_7 m_8
    #
    # Value of Dense(1): m_1 x_1 + m_2 x_2 + ... + m_8 x_8 + c. We have 8 weights
    # and one bias, so in total we have 9 total trainable variables.
    #
    # You may notice that the dimensions of the neuron layers have "None" in
    # them. In the inputs, this is the "batch size" of the model. Since ours is
    # None, it means it is unspecified. However, with large datasets, it is not
    # possible to train the entire dataset at once; so, we train the model with
    # batches of our data. The reason it is not possible to train entire datasets
    # at once is due to the fact that if a dataset is several terabytes (TB), and
    # the computer you are working from only has RAM on the order of tens to
    # hundreds of gigabytes, then you cannot store and compute all of the data
    # all at once. In this toy example, the data is very small so we wouldn't
    # actually need to run it in batches, but for the sake of education, we will.
    # A topic to come back to and subject of research is that large batch sizes
    # will make a model converge quicker, however large batch sizes can cause
    # overfitting.

    batch_size = 5
    model = tf.keras.Sequential([
        InputLayer(input_shape = (batch_size, 8)),
        normaliser, # Normalisation layer - has output shape (None, 8)
        Dense(1),   # Single dense neuron layer - has output shape (None, 1)
    ])
    # model.summary() # View the model summary
    # tf.keras.utils.plot_model(model, show_shapes = True) # View model layer plot
    #
    #
    # Model Error Analysis:
    # --------------------------
    # We want to see how well our model, the best fit line, compares with the
    # actual data. Remember that for each input point in X, the model produces
    # an estimate y_p which is the predicted value. Putting in existing values
    # from our data, we can see how the model compares to the real-world data.
    # Which we define to be y_a:
    #
    #              _     y          y = m x + c
    #              | y_a |---- .  ./.
    #  abs. Error  |     |     |. / .
    #  |y_a - y_p| |     |  .  | / .
    #              |     |    .|/
    #              | y_p |-----/  .
    #              ‾     |. . /|      .
    #                    | . / |    .
    #                    |  / .| .
    #                    | /.  |   . <-- (1)
    #                    |/  . |
    #                    | .   |                  X
    #                    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    # Note that we an absolute error |y_a - y_p| between the predicted and the
    # real-world values. What we want to do is minimise the error as much as
    # possible. We can customise the error function to what ever we like. We
    # can choose a loss function to be Loss(y_a, y_p) = (y_a - y_p) ^ 2. An
    # error function measures deviations from an observable value (y_a) from a
    # predicted value (y_p), and a loss function is an operation on the error
    # to quantify the severity of a loss of a particular value, a "cost of being
    # wrong" so to speak which is why it is also written as a cost function. In
    # some cases, absolute errors are not used and errors with signs, either
    # positive, negative, or zero, are used and can have different meanings in a
    # loss/cost function.
    #
    # The function Loss(y_a, y_p) = (y_a - y_p) ^ 2 is known as the "Mean Square
    # Error" Loss function. In our model, we want to find all the errors across
    # all points between the real and predicted in our dataset, and find the
    # average. We ca do this with the tf.keras.losses.MeanSquaredError() method.
    # There are many other types of loss functions in the documentation. For
    # example, the mean abolute error takes the absolute difference between y_a
    # and y_p and finds the mean across all data points. This has an advantage
    # in that large errors do not have as large losses compared to the mean
    # squared error loss function, since the error is squared in the latter.
    # This means that the gradient of the curve (given by the m's) is not heavily
    # influenced by outliers data points such as (1) in the ASCII chart above.
    #
    # We can use the mean squared error and mean absolute error more intelligently.
    # We'll use the mean square error for datapoints close to the regression line
    # and then the mean absolute error for outlier points. This technique is called
    # the "Huber Loss" function. We can use it via the tf.keras.losses.Huber()
    # method, and will switch between using MSE and MAE if the error is greater
    # than a value delta.

    # Compile the model with a specific loss function
    model.compile(
        loss = MeanAbsoluteError
        # loss = MeanSquaredError()
        # loss = Huber()
    )


    # Training and Optimisation:
    # --------------------------
    # In our graph above, we want to find the values of the weights and bias m
    # and c such that the regression line best fits the data, determined by the
    # lowest mean absolute/squared error (or Huber method or some other way).
    # There are uncountably infinitely many combinations for weights and biases.
    # We first initialise random weights and biases, although there are ways of
    # seeding reasonable values to make the model converge faster. The method
    # currently to minimise error used is called "Stochastic Gradient Descent".
    # Since the values of the cost function form a surface over our N = 8
    # dimensional space, we will want to find the minima of this surface, which
    # are the lowest points on this surface. In a one dimensional case, think
    # some polynomial like y = x^2. Its minimal value is found by computing its
    # (partial) derivative ∂y/∂x = ∂/∂x [x^2] = 2x. For higher degree polynomials
    # this will give the locations of minimal values but not neccessarily the
    # smallest minima.
    # 
    # Instead, what we can do is to step either left or right depending on the
    # sign on the derivative and make these steps proportional to the magitude
    # of the gradient. So, very negative gradients will imply that a local
    # minimum is to the right, and very positive gradients will imply a left
    # local minima. Doing this repeatedly will ensure convergence to a minima.
    # Note that finding a local minima is generally very doable but finding the
    # global minima is very difficult.
    #
    # In stochastic gradient descent on N-dimensional hypersurfaces, we want to
    # find the direction which decreases the cost function the fastest. From
    # multivariable calculus, the gradient function tells us the direction which
    # increases the function most quickly, so the negative of this will be the
    # direction of the function that decreases it most quickly. It is denoted
    # by -∇[ ]. So, we compute ∇[Loss(y_a, y_b)], step in the direction of
    # -∇[Loss(y_a, y_b)], and repeat until we approach a local minima. The
    # components of -∇[Loss(y_a, y_b)] tell us which features need increasing
    # or decreasing.
    
    
# ============================================================================ #
# Car Price Prediction - Code End                                              |
# ============================================================================ #
