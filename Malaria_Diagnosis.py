#!/usr/bin/env python3
# ============================================================================ #
# Malaria Diagnosis - Frederick T. A. Freeth                        25/08/2023 |
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
    #
    # Image data is represented by a 2d array with an RGB colour space.
    
    
    
# ============================================================================ #
# Malaria Diagnosis - Code End                                                 |
# ============================================================================ #
