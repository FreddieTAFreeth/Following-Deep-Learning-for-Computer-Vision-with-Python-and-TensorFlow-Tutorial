#!/usr/bin/env python3
# ============================================================================ #
# Malaria Diagnosis - Frederick T. A. Freeth                        29/08/2023 |
# ============================================================================ #
# Following https://www.youtube.com/watch?v=IA3WxTTPXqQ.

import csv
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # The Task:
    # ---------------------------
    # In 2021, almost half of the World's population was at risk of malaria. In
    # that year there were an aestimated 247 million cases of malaria worldwide.
    # The WHO (World Health Organisation) African Region experiences a massive
    # disproportion compared to the rest of the world. In 2021, the Region is
    # home to 95% of malaria cases and 96% of malaria deaths, of which there
    # 619,000. Also, children under 5 accounted for approximately 80% of all
    # deaths in the Region. Source (accessed 24/08/2023): 
    # https://www.who.int/news-room/fact-sheets/detail/malaria
    #
    # In this file, we will build a machine learning model based on convolutional
    # neural networks (CNNs) that aim to diagnose malaria. The data we will use
    # are microscopy photos of cells. These cells are gathered via puncture of a
    # fingertip, and blood is smeared out on a slide. We can have two types of
    # smears: thin or thick. Our dataset is concerned with only thin smears since
    # these are used in the differentiaton of leukocytes. A thick blood smear is
    # used for the diagnosis of blood protozoan parasites and blood abnormalities.
    # Each slide photograph will contain many cells. We will need to analyse each
    # cell seperately by extracting it from the image of th entire slide.
    #
    # Our model takes in as input
    # a photo of the cells on the slide, and as output gives a binary decision of
    # whether the person is infected with malaria or not:
    #
    #     Inputs                         Outputs
    #     ‾‾‾‾‾‾                         ‾‾‾‾‾‾‾
    #   [ Cell   ]     [       ]     [ Host Status: ]
    #   [ Photo- ] --> [ Model ] --> [ infected or  ]
    #   [ graph  ]     [       ]     [ not infected ]


    # Data Preparation:
    # ---------------------------
    # Image data is represented by a 2d array of pixels each with an RGB colour
    # space. So, image tensors have shape (IMAGE_HEIGHT, IMAGE_WIDTH, 3). The
    # colours of the pixels are an array [RED, GREEN, BLUE] where the colour
    # channels range between 0 to 255. We can normalise them as well by division
    # by 255. In the greyscale format, the number of channels is 1 so just has
    # shape (IMAGE_HEIGHT, IMAGE_WIDTH, 1), a 2d tensor.
    #
    # Our dataset is included in the TensorFLow library. It contains a total of
    # 27,558 cell images with equal counts of parasitized and uninfected cells
    # from the thin blood smear slide images.
    # https://www.tensorflow.org/datasets/catalog/malaria

    # We can use the tsdf.load() method to load the data in via the Dataset API.
    # Now, malaria_data is a dictionary with two entries 'train' and 'types'.
    malaria_data, malaria_data_info = tsdf.load(
        name = "malaria",     # The name of the dataset
        with_info = True,     # Imports information about the dataset
        as_supervised = True, # Returns images as labeled tuple (input, label)
        shuffle_files = True, # This shuffles the images in the file
        split = ["train"]     # Which splits of the data to load
    )

    # Break up the dataset into the training, testing, and validation datasets.
    # You can use the Dataset TensorFlow API (using take, skip, etc) but this
    # will suffice.
    # Define the proportion of each dataset you want
    N = len(malaria_data) # Number of cell images
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

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print(X_val.shape, y_val.shape)
    
    
# ============================================================================ #
# Malaria Diagnosis - Code End                                                 |
# ============================================================================ #
