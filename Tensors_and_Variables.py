#!/usr/bin/env python3
# ============================================================================ #
# Tensors and Variables - Frederick T. A. Freeth                    08/08/2023 |
# ============================================================================ #
# Following https://www.youtube.com/watch?v=IA3WxTTPXqQ.

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    # Notes:
    # ------

    # Tensor Dimension:
    # ---------------------------
    # Tensors are multidimensional arrays.
    # The shape of the tensor [8] is 0.
    # The shape of the tensor [2 0 -3] is (3,).
    # The shape of the tensor [[1 2 0], [3 5 -1], [1 5 6], [2 3 8]] is (4, 3).
    # The shape of the tensor [[[1 2 0], [3 5 -1]], [[1 5 6], [2 3 8]]] is (2, 2, 3).
    # Note how the length of the tuple is the depth of the tensor.

    # Initialisation and Casting
    # ---------------------------
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


    # Indexing Tensors:
    # ---------------------------
    # Just like in pretty much all programming langages, indexing is done with
    # square brackets. This also returns a tensor. To recap, python indexing is
    # done like: myTensor[min : max + 1 : step]. Similar to R, you can extract
    # rows and columns by specifying indicies: myTensor[rows, columns].
    # print(tensor_2d)
    # print(tensor_2d[:,1]) # Second column
    # print(tensor_2d[3, :]) # Last row

    # Indexing scales up for arbitrary tensor sizes.
    # print(tensor_3d)
    # print(tensor_3d[:,:,2])


    # Tensorflow Math Functions:
    # ---------------------------
    # https://www.tensorflow.org/api_docs/python/tf/math.
    # See the documentation tf.math for more details. Most of the functions work
    # element wise across tensors. Contained therein are abs(), trigonometric,
    # and many other mathematical functions. TensorFlow can also work with
    # complex data types. The tf.abs() method will compute the magnitude of a
    # complex number, not just turning components positive:
    z = tf.constant([-3.141 + 2.718j])
    z_abs = tf.abs(z) # if z = a + bi -> |z| = sqrt(a^2 + b^2)

    # Arithmetric of tensors is element-wise if they have the same shape
    T1 = tf.constant([1, 2, 3, 4, 5], dtype = tf.float32)
    T2 = tf.constant([1, 0, -3, 5, 2], dtype = tf.float32)
    # print(tf.add(T1, T2))
    # print(tf.subtract(T1, T2))
    # print(tf.multiply(T1, T2))
    # print(tf.divide(T1, T2)) # Produces an inf value in element 2
    # print(tf.math.divide_no_nan(T1, T2)) # Replaces inf by zeroes with 0.
    # print(tf.sqrt(T1))

    # Broadcasting tensors. If one tensor is smaller than another, the smaller
    # one is scaled-up and then arithmetric can be done element-wise like in R.
    T3 = tf.constant([5], dtype = tf.float32)
    T4 = tf.constant([[7], [8], [9]], dtype = tf.float32)
    # print(tf.add(T1, T3))
    # print(tf.add(T1, T4)) # Creates a (3, 5) tensor instead.
    # Other functions such as tf.math.maximum and tf.math.minimum will find the
    # minimum and maximum element-wise comparisons between tensors and supports
    # broadcasting semantics. The argmax and argmin methods will return the index
    # of the maximum values in a tensor.
    # print(tensor_3d)
    # print(tf.math.argmax(input = tensor_3d, axis = None, output_type = tf.dtypes.int64, name = None))
    # print(tf.math.argmin(tensor_3d, axis = None, output_type = tf.dtypes.int64, name = None))
    # When axis  = 0, it will fix across rows and compare across columns. If the
    # axis = 1, we operate across rows of the tensor.

    # We can raise elements of one tensor to powers of elements to another tensor.
    # print(tf.pow(T1, T3))

    # The method tf.math.reduce_sum() find the sum of all elements across the
    # dimensions of a tensor.
    # print(tf.math.reduce_sum(
    #     input_tensor = tensor_3d, axis = None, keepdims = False, name = None
    # ))
    # tf.math.reduce_max and tf.math.reduce_min get the single max or min values
    # in the tensor. tf.math.reduce_mean() operates exactly the same, and you can
    # also specify the axes you wat to operate across. The keepdims argument in
    # the reduce family of methods the reduced dimensions with length 1.
    # For more complex tensors, more care is needed to ensure correct broadcasting.

    # The method tf.math.sigmoid() computes the sigmoid element-wise.
    sigmoid_tensor_1d = tf.math.sigmoid(x = tf.cast(tensor_1d, dtype = tf.float16), name = None)

    # The method tf.math.top_k() find the values and indicies for the k largest
    # entries for the last dimension.
    top_k_tensor_1d = tf.math.top_k(input = tensor_1d, k = 2, sorted = True, name = None)
    
# ============================================================================ #
# Tensors and Variables - Code End                                             |
# ============================================================================ #
