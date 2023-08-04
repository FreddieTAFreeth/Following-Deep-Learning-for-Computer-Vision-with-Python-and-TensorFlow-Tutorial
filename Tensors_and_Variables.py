#!/usr/bin/env python3
# ============================================================================ #
# Tensors and Variables - Frederick T. A. Freeth                    04/08/2023 |
# ============================================================================ #
# Following https://www.youtube.com/watch?v=IA3WxTTPXqQ.

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    # Notes:
    # ------

    # Tensor Dimension:
    # Tensors are multidimensional arrays.
    # The shape of the tensor [8] is 0.
    # The shape of the tensor [2 0 -3] is (3,).
    # The shape of the tensor [[1 2 0], [3 5 -1], [1 5 6], [2 3 8]] is (4, 3).
    # The shape of the tensor [[[1 2 0], [3 5 -1]], [[1 5 6], [2 3 8]]] is (2, 2, 3).
    # Note how the length of the tuple is the depth of the tensor.

    # Create tensors
    tensor_0d = tf.constant(8) # prints as tf.Tensor(8, shape=(), dtype=int32)
    tensor_1d = tf.constant([2, 0, -3]) # tf.Tensor([ 2  0 -3], shape=(3,), dtype=int32)

    # tf.Tensor([[ 1  2  0] [ 3  5 -1] [ 1  5  6] [ 2  3  8]], shape=(4, 3), dtype=int32)
    tensor_2d = tf.constant([
        [1, 2, 0], [3, 5, -1], [1, 5, 6], [2, 3, 8]
    ])

    #tf.Tensor([[[ 1  2  0] [ 3  5 -1]] [[ 1  5  6] [ 2  3  8]]], shape=(2, 2, 3), dtype=int32)
    tensor_3d = tf.constant([
        [[1, 2, 0], [3, 5, -1]], [[1, 5, 6], [2, 3, 8]]
    ])
    # ... and so on. The methods .ndim and shape give the dimensions and shape.
    # Specifying the dtype argument in tf.constant allows us to change the data
    # type. Different datatypes use different amounts of memory just like in C.
    # https://www.tensorflow.org/api_docs/python/tf/dtypes. If you have some
    # tensor that is in integer type, using tf.cast() we can cst to another type.
    tensor_2d_cast = tf.cast(tensor_2d, dtype = tf.float16)

    # You an have boolean and string tensors. It is possible to convert numpy
    # arrays to tensors using tf.convert_to_tensor().

    # Using tf.eye(), we can create an identity tensor. If num_columns != None,
    # then it creates a square matrix. Also, batch_shape allows you to specify
    # the depth of the matrix.
    id_tensor = tf.eye(
        num_rows = 3, num_columns = 4, batch_shape = [2,],
        dtype = tf.dtypes.float16, name = None
    )

    # The method tf.fill() allows you to create tensor filled with specified
    # values. The tf.ones() method fills with just 1, and tf.zeros() with 0.
    fill_tensor = tf.fill(dims = [2, 3], value = 2, name = None)
    ones_tensor = tf.ones(shape = [2, 2, 3], name = None)
    zeros_tensor = tf.zeros(shape = [3, 4, 3], name = None)

    # The tf.ones_like() method will create a tensor of all ones that has the
    # shape as the input tensor. E.g. tf.ones([1 2 3]) becomes [1 1 1].
    ones_tensor = tf.ones_like(tensor_2d, dtype = None, name = None)

    # The tf.shape() method returns a tensor containing the shape of the input.
    tensor_3d_shape = tf.shape(input = tensor_3d, out_type = tf.dtypes.int32, name = None)
    # This produces tf.Tensor([2 2 3], shape=(3,), dtype=int32).

    # To find the rank of a tensor, use tf.rank()
    tensor_3d_rank = tf.rank(input = tensor_3d, name = None) # = 3

    # The size of a tensor is found via tf.size() and is the number of elements.
    tensor_3d_size = tf.size(input = tensor_3d, out_type = tf.dtypes.int32, name = None)

    # Create a tensor consisting of random numbers. Using tf.random.set_seed(),
    # you can set a seed for reproducible results.
    rand_normal_tensor = tf.random.normal(
        shape = [2, 3, 4], mean = 0.0, stddev = 1, dtype = tf.dtypes.float32
    )
    rand_uniform_tensor = tf.random.uniform(
        # You must specify a maxval if using integer dtypes.
        shape = [2, 2], minval = 0, maxval = 1, dtype = tf.dtypes.float32,
        seed = None, name = None
    )

# ============================================================================ #
# Tensors and Variables - Code End                                             |
# ============================================================================ #
