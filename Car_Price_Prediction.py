#!/usr/bin/env python3
# ============================================================================ #
# Car Price Prediction - Frederick T. A. Freeth                     29/08/2023 |
# ============================================================================ #
# Following https://www.youtube.com/watch?v=IA3WxTTPXqQ.

import csv
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    # The Task:
    # ---------------------------
    # We want to predict the price of second-hand cars given several input
    # features. Owners of these cars will specify:
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
    # NOTE: Only the final version of the model at the very end has been left
    # uncommented, and uses all the combined knowledge in this script. Now,
    # however, it is reccomended to read this sequentially and copy and paste
    # the bits of code into another file, and modify in-place as you read this
    # file.
    #
    #
    # The Model:
    # ---------------------------
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
    # Note: I have removed the "on road old" and the "on road now" features of
    # the dataset.
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
    # We can now read in the model data, and converting it to a tensor.

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
    N = len(X) # Number of data points (essentially the number of cars)
    

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
    # is why we need the normalisation layer. Normalising the inputs X can speed
    # up the training process. We normalise by subtracting the mean and dividing
    # by the standard deviation: X ^ {tilde} = (X - μ) / (σ ^ 2). The dense layer
    # is a fully connected layer, meaning that each neuron from the previous layer
    # is connected to a neuron in the dense layer forming a bijection between the
    # two layers. What it does is that that it takes an input x, multiply it by a
    # weight, and then add a bias. So, mx + c = y_p which is our predicted value
    # of y. For M = 8 features, we connect 8 neurons from the input layer into
    # the normalisation layer which has the same shape as the input, so we can
    # normalise each value. From here, these normalised inputs pass into the
    # dense layer which has an output shape of 1, since we want to know what a
    # predicted car's price is. This predicted price is a singular number, which
    # is why the dense layer must have an output of shape of 1.
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

    # batch_size = 5
    # normaliser = tf.keras.layers.Normalization() # Init normaliser
    # normaliser.adapt(X) # Find the mean and standard deviation of each column

    # Build the neural network model
    # model = tf.keras.Sequential([
    #     tf.keras.layers.InputLayer(input_shape = (8, )),
    #     normaliser, # Normalisation layer - has output shape (None, 8)
    #     tf.keras.layers.Dense(1) # Single dense neuron layer - has output shape (None, 1)
    # ])
    # model.summary() # View the model summary
    # tf.keras.utils.plot_model(model, show_shapes = True) # View model layer plot

    
    # Model Error Analysis:
    # ---------------------------
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

    # Compile the model with a specific loss function. In tf.keras.optimizers,
    # there are different optimisation algorithms to choose from. We will
    # discuss optimisers in the nxt section. The most common optimiser is the
    # Adam optimiser, which is a stochastic gradient descent method based on
    # adaptive estimation of first and second-order moments. It is computationally
    # efficient and has little memory requirement.
    # model.compile(
        # optimizer = tf.keras.optimizers.SGD(), # Stochastic Gradient Descent
    #     optimizer = tf.keras.optimizers.Adam(learning_rate = 1),
        # loss = tf.keras.losses.MeanAbsoluteError(),
        # loss = tf.keras.losses.MeanSquaredError(),
    #     loss = tf.keras.losses.Huber(),
    #     metrics = tf.keras.metrics.RootMeanSquaredError() # More on this in Performance Measurement!
    # )


    # Training and Optimisation:
    # ---------------------------
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
    # by -∇[ ]. So, we compute ∇[C(W)] where C(W) is a cost function of the
    # weights and biases. Telling the computer is is doing badly is not helpful,
    # so we need to tell the computer how to change the weights and biases to
    # improve, i.e., reduce, the error and by extension the cost.
    #
    # We step in the direction of -∇[C(W)], and repeat until we approach
    # a local minima. The components of -∇[C(W)] tell us which features need
    # increasing or decreasing and how to reduce the value of the cost function
    # the fastest. The formula for stochastic gradient descent is given by the
    # equation w := w - η ∇[C(W)] for η the "learning rate". In 1 dimension,
    # this is analagous to finding the gradient of a curve, and then stepping
    # towards the downhill direction as eluded to above. This process will be
    # repeated until we achieve sufficient minimisations in losses.
    #
    # Backpropagation is the way we can calculate this gradient efficently,
    # since in a a high dimensional space, ∇[C(W)] is very difficult to find.
    # This involves modifying weights and biases of layers of the neural network
    # to reduce the value of the cost function. The number of epochs is the
    # times we will perform the gradient descent step. The fit method in the
    # tf.keras.Model class allows us to train our neural network.
    
    # Train the neural network with batch_size batches and 100 epochs
    # model_fitting = model.fit(
    #     x = X, y = y, batch_size = batch_size, epochs = 100, verbose = 0
    # )
    # Setting verbose = 1 will give a printout of the progress of the different
    # epochs, plus information about losses too (more on this shortly).

    # We can plot the values of the losses generated when fitting the model.
    # Changing the learning_rate parameter will change how fast the model will
    # converge.
    # plt.plot(model_fitting.history["loss"])
    # plt.title("Model Losses")
    # plt.ylabel("Model Loss")
    # plt.xlabel("Epoch")
    # plt.legend(["Loss"])
    # plt.show()

    # Performance Measurement:
    # ---------------------------
    # We want to quantify how well the model performs. A common way used to
    # evaluate model performance is the "Root Mean Square Error" (RMSE). Using
    # a performance measurement, we can compare two model's performance on the
    # same dataset. We can configure our metric when we compile our model with
    # the "metric" parameter. This has been done above. When we train our model,
    # the losses and the metric will both be printed. We can now plot the RMSE.

    # plt.plot(model_fitting.history["root_mean_squared_error"])
    # plt.title("Model Performance")
    # plt.ylabel("Root Mean Squared Error")
    # plt.xlabel("Epoch")
    # plt.legend(["RMSE"])
    # plt.show()

    # Another method for evaluating a TensorFlow model is using the evaluate()
    # method from the tf.keras.Model class. It returns the loss value and metrics
    # values for the model in test mode.
    
    # model.evaluate(X, y)


    # Validation and Testing:
    # ---------------------------
    # We want to be able to test whether our model actually work. At the moment,
    # our model is trained on a narrow dataset, and may be fitted to values in
    # our data. Now, what if we want to supply in values the model hasn't seen
    # before? How will it perform? This is why we we validate and test the model.
    # What we want is for new, unseen data to produce reasonable outputs. What
    # we can do is break up our dataset into two halves: once for training, and
    # then one for testing. Out of our 1000 cars, we can supply 800 for training
    # and then the last 200 for validation and testing. When we shuffled the
    # dataset at the start of the script, this helps us make sure there are no
    # underlying biases in how the dataset is structured or in the order of
    # how it was collected or stored.
    #
    # Now, supposing we have some insanely massive dataset. We don't want to
    # wait until it has finished training before we can validate it. So, while
    # it is training, we want to see how it performs on data it hasn't seen.
    # This data is called the "Validation Set". The data we used to train the
    # model is the "Training Set", and set of data used to test the model after
    # it has been trained is the "Testing Set". We now split up our dataset into
    # the training set, the testing set, and the validation set.

    # Define the proportion of each dataset you want
    TRAIN_PROPORTION = 0.8
    TESTING_PROPORTION = 0.1
    VALIDATION_PROPORTION = 0.1
    assert TRAIN_PROPORTION + TESTING_PROPORTION + VALIDATION_PROPORTION == 1

    # Create the IDs of each set
    train_IDs      = np.arange(0, round(N * TRAIN_PROPORTION))
    test_IDs       = np.arange(len(train_IDs), len(train_IDs) + round(N * TESTING_PROPORTION))
    validation_IDs = np.arange(len(test_IDs),  len(test_IDs)  + round(N * VALIDATION_PROPORTION))
    assert len(train_IDs) + len(test_IDs) + len(validation_IDs) == N

    # Define the training, test, and validation sets
    X_train = X[train_IDs[0]:train_IDs[-1]]
    y_train = y[train_IDs[0]:train_IDs[-1]]
    X_test  = X[test_IDs[0]:test_IDs[-1]]
    y_test  = y[test_IDs[0]:test_IDs[-1]]
    X_val   = X[validation_IDs[0]:validation_IDs[-1]]
    y_val   = y[validation_IDs[0]:validation_IDs[-1]]

    # Now, we go back and reoeat this process, but using our different data sets:
    batch_size = 5
    # normaliser = tf.keras.layers.Normalization()
    # normaliser.adapt(X_train)
    
    # We may also modify the neural network model. Doing the stuff below with the
    # first version of the model results in the model being very poor. Skip ahead
    # to the end with the old version to try it yourself! Some corrective measures
    # are to make the model more complex. We can create some more dense "hidden"
    # layers at the cost of extra computing performance. They work in a similar
    # way in that all neurons from the previous layers are all connected to the
    # next layer, excpet there are now far more connections between layers. It
    # looks like the following (however connections are removed since drawing in
    # ASCII is difficult ;) just imagine each neuron from neuron is connected to
    # the neurons in the next).
    #
    #  x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8
    #   ○   ○   ○   ○   ○   ○   ○   ○   Input Layer
    #
    #                 |
    #                 V
    #
    #   ○   ○   ○   ○   ○   ○   ○   ○   Normalisation Layer
    #
    #                 |
    #                 V
    #
    #             ○   ○   ○             Dense(3) Layer
    #
    #                 |
    #                 V
    #
    #           ○   ○   ○   ○           Dense(4) Layer
    #
    #                 |
    #                 V
    #
    #             ○   ○   ○             Dense(3) Layer
    #
    #                 |
    #                 V
    #
    #                 ○                 Dense(1) Layer
    #
    # We can also change how neurons activate in the layers as this will also
    # influence the learning process. Early functions were the sigmoid activation
    # functions but more state of the art methods involve ones such as the ReLU
    # (Recified Linear Unit) and Tanh functions. For a given connection between
    # two neurons, we had this y = m x + c type equation. Well, we actually pass
    # this through one of the "squishification" functions like the sigmoid or the
    # ReLU functions, to map the value of y = m x + c to a number between 0 and 1.
    # Values of 0 are inactive neurons and values close to 1 are active neurons.
    # The properties of these activations functions (meaning broadly the
    # mathematical properties of them) determine, in collaboration with the
    # weights and biases, determine the tendancy for a neuron to activate.
    # We specify the activation functions within the layers themselves:
    
    # model = tf.keras.Sequential([
    #     tf.keras.layers.InputLayer(input_shape = (8, )),
    #     normaliser,
    #     tf.keras.layers.Dense(128, activation = "relu"),
    #     tf.keras.layers.Dense(256, activation = "relu"),
    #     tf.keras.layers.Dense(128, activation = "relu"),
    #     tf.keras.layers.Dense(1)
    # ])

    # And don't forget to re-compile this new model!
    # model.compile(
    #     optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1),
    #     loss = tf.keras.losses.Huber(),
    #     metrics = tf.keras.metrics.RootMeanSquaredError()
    # )
    
    # We can now re-fit the model, but instead we can use the validation data we
    # have defined and fit the model to the training data. Instead of manually
    # breaking apart the dataframe like we have done, you can use the "shuffle"
    # (Boolean) and "validation_split" (a number between 0 and 1) arguments and
    # this will do the data shuffling and partitioning for you which is helpful.
    # To also help with the model performance, we can increase the number of
    # epochs too. 
    
    # model_fitting = model.fit(
    #     x = X_train, y = y_train, batch_size = batch_size, epochs = 256,
    #     validation_data = (X_val, y_val), verbose = 0
    # )
    # Running the above with verbose = 1 will now also print the "val_loss", the
    # validation loss, and "val_root_mean_squared_error" which is the RMSE against
    # the validation data across the different epochs. We can now plot these.

    # Plot for losses
    # plt.plot(model_fitting.history["loss"])
    # plt.plot(model_fitting.history["val_loss"])
    # plt.title("Model Losses")
    # plt.ylabel("Loss")
    # plt.xlabel("Epoch")
    # plt.legend(["Loss", "Validation Loss"])
    # plt.show()

    # Plot for the error
    # plt.plot(model_fitting.history["root_mean_squared_error"])
    # plt.plot(model_fitting.history["val_root_mean_squared_error"])
    # plt.title("Model Performance")
    # plt.ylabel("Root Mean Squared Error")
    # plt.xlabel("Epoch")
    # plt.legend(["RMSE", "Validation RMSE"])
    # plt.show()

    # You'll notice that the losses in the training data are lower than in the
    # validation set. This is normal since the values in the training set the
    # model has already seen! However, if the model performs really well in the
    # training data but really poorly in the validation data, then it is said
    # that the model is "over-fitting".

    # And we can also evaluate our model on the training and test sets, too:
    # model.evaluate(X_train, y_train)
    # model.evaluate(X_test, y_test)

    # Let's test the model for every value from the testing set:
    # model.predict(X_test)

    # You can also test a singular datapoint too (after expanding the dimension):
    # car_price_prediction = model.predict(tf.expand_dims(X_test[0], axis = 0))

    # This tells us that for the first car in the testing set, it predicts the
    # price given by the output of model.predict(). How does it compare? Well,
    # having the basic network of the input layer, the normalisation layer, and
    # then the Dense(1) layer fits extremely poorly. The prices are way lower
    # than what is expected - a situation known as "under-fitting".
    # car_price_actual = y_test[0]

    # Final result:
    # price_difference = car_price_prediction - car_price_actual
    # print(f"Model Prediction: ${car_price_prediction}")
    # print(f"Actual Price: ${car_price_actual}")
    # print(f"Car Price Difference: ${price_difference}")

    # If not done so already, go back and use the very first version of the
    # model with just the input, normalisation, and Dense(1) layer. Notice how
    # badly this performs!


    # Faster Ways To Import Data:
    # ---------------------------
    # Now, in machine learning, it is common for datasets to be massive. How can
    # we import massive datasets faster? The tf.data.Dataset class has methods
    # that can help us. Using the from_tensor_slices() method, we can import our
    # data much faster. This speed increase is really apparent on really massive
    # datasets.
    training_data   = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validation_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_data       = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # We can also shuffle the data as well using the shuffle() method. The method
    # parameter "buffer_size" is the number of samples which are randomised. The
    # batch() method creates the batches given. Then, prefetch() pre-loads data,
    # at the cost of more memory usage, so that once a current batch has finished
    # processing, the next batch has already been loaded and prepared whilst the
    # model is training the current batch. This often improves latency and
    # data throughput.
    #
    # |---------------------------------------------------------> Time
    #           Loading           Loading           Loading
    # |||||||||---------|||||||||---------|||||||||---------||||||||| --> Slow!
    # Training          Training          Training          Training
    #
    #  Loading   Loading   Loading   Loading
    # --------- --------- --------- ---------
    # |||||||||-|||||||||-|||||||||-||||||||| --> Faster since data for the next
    # Training  Training  Training  Training      training block was pre-loaded
    #
    # The diagram above illustrates how the model will train the batches, and
    # then pre-loads the data for the next batch.
    
    training_data = training_data.shuffle(
        buffer_size = 8, reshuffle_each_iteration = True
    ).batch(32).prefetch(tf.data.AUTOTUNE)

    validation_data = validation_data.shuffle(
        buffer_size = 8, reshuffle_each_iteration = True
    ).batch(32).prefetch(tf.data.AUTOTUNE)

    test_data = test_data.shuffle(
        buffer_size = 8, reshuffle_each_iteration = True
    ).batch(32).prefetch(tf.data.AUTOTUNE)

    # You can print the data in the batches as well:
    # for x, y in training_data: print(x, y)

    # We don't need to make any changes to the model, but we include it below:

    # Re-initialise the normalisation layer
    normaliser = tf.keras.layers.Normalization()
    normaliser.adapt(X_train)
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape = (8, )),
        normaliser,
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dense(256, activation = "relu"),
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dense(1)
    ])

    # Model compilation
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1),
        loss = tf.keras.losses.Huber(),
        metrics = tf.keras.metrics.RootMeanSquaredError()
    )
    
    # Fitting the model
    model_fitting = model.fit(
        training_data, batch_size = batch_size, epochs = 256,
        validation_data = (X_val, y_val), verbose = 1
    )

    # Plot for losses
    plt.plot(model_fitting.history["loss"])
    plt.plot(model_fitting.history["val_loss"])
    plt.title("Model Losses")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Loss", "Validation Loss"])
    plt.show()

    # Plot for the error
    plt.plot(model_fitting.history["root_mean_squared_error"])
    plt.plot(model_fitting.history["val_root_mean_squared_error"])
    plt.title("Model Performance")
    plt.ylabel("Root Mean Squared Error")
    plt.xlabel("Epoch")
    plt.legend(["RMSE", "Validation RMSE"])
    plt.show()

    # Model evaluation
    model.evaluate(X_train, y_train)
    model.evaluate(X_test, y_test)
    car_price_predictions = list(model.predict(X_test)[:, 0])
    car_price_actual = list(y_test[:, 0])

    # Plotting the real and predicted car prices:
    plt.plot(car_price_predictions)
    plt.plot(car_price_actual)
    plt.title("Model Prediction")
    plt.ylabel("Car Price, $")
    plt.xlabel("Car")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()

# ============================================================================ #
# Car Price Prediction - Code End                                              |
# ============================================================================ #
