#!/usr/bin/env python3
# ============================================================================ #
# Tensors and Variables - Frederick T. A. Freeth                    12/08/2023 |
# ============================================================================ #
# Following https://www.youtube.com/watch?v=IA3WxTTPXqQ.

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    # Notes:
    # ---------------------------

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
    # ----------------------------
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
    # ----------------------------
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
    # entries for the last dimension row-wise.
    top_k_tensor_1d = tf.math.top_k(input = tensor_1d, k = 2, sorted = True, name = None)


    # Linear Algebra Operations:
    # ----------------------------
    # https://www.tensorflow.org/api_docs/python/tf/linalg
    # More information and details in the documentation.

    # Matrix multiplications is done via tf.linalg.matmul()
    M1 = tf.constant([
        [1, 2, 3],
        [4, 5, 6]
    ])
    
    M2 = tf.constant([
        [ -7,  8],
        [ -9, 10],
        [-11, 12]
    ])

    M3 = tf.constant([
        [1,    2,   3,    4],
        [0, -1/2, 1/3, -1/4],
        [5,   -6,   7,   -8],
        [1,    0,   1,    0]
    ])

    # Matrix multiplication with Tensorflow. If one matrix needs to be transposed,
    # you can specify which one via the function arguments.
    matrix_product = tf.linalg.matmul(
        a = M1, b = M2, transpose_a = False, transpose_b = False,
        adjoint_a = False, adjoint_b = False, a_is_sparse = False,
        b_is_sparse = False, output_type = None, name = None
    )

    # Matrix transpose:
    M1_transpose = tf.transpose(M1)

    # Matrix adjoints:
    M1_adjoint = tf.linalg.adjoint(M1)

    # The tf.linalg.band_part() method sets everything outside a central band in
    # each innermost matrix to zero. Special cases written in the documentation:
    # tf.linalg.band_part(input, 0, -1) ==> Upper triangular part.
    # tf.linalg.band_part(input, -1, 0) ==> Lower triangular part.
    # tf.linalg.band_part(input, 0, 0) ==> Diagonal.

    # Te method tf.linalg.cholesky() find the Cholesky decomposition of one or
    # more square matricies.
    M3_Cholesky = tf.linalg.cholesky(input = M3, name = None)

    # The corss product is calculated from tf.linalg.cross(). They must have the
    # same shape.
    M1_M1_cross = tf.linalg.cross(a = M1, b = M1, name = None)

    # Inverses of square matricies are found via tf.linalg.inv():
    M3_inverse = tf.linalg.inv(input = M3, adjoint = False, name = None)

    # Matrix singular value decomposition (SVD):
    s_M3, u_M3, v_M3 = tf.linalg.svd(tensor = M3, full_matrices = False, name = None)

    # The method np.einsum() is computed as follows. The value i is the number
    # of rows of M1, j the number of columns in M1 which is equal to the number
    # of rows in M2. Lastly, k is the number of columns of M2.
    matrix_product = np.einsum("ij, jk -> ik", M1, M2)

    # We can use the einsum syntax to do Hadamard (element-wise) multiplication
    # of matricies with the same shape:
    matrix_product = np.einsum("ij, ij -> ij", M1, M1)

    # Matrix transpose via einsum:
    matrix_transpose = np.einsum("ij -> ji", M1)

    # For b = batchsize and two tensors of shape (2, 3, 4) and (2, 4, 5), we
    # create the tensor of shape (2, 3, 5) va a batch multiplication using
    # numpy and einsum. However, einsum is more explicit and easier to debug.

    T1 = np.array([
        [[2,  6, 5, 2],
         [2, -2, 2, 3],
         [1, 5, 4,  0]],
         
        [[1, 3, 1, 22],
         [0, 2, 2,  0],
         [1, 5, 4,  1]]
    ])
    
    T2 = np.array ([
        [[2, 9, 0,  3, 0],
         [3, 6, 8, -2, 2],
         [1, 3, 5,  0, 1],
         [3, 0, 2,  0, 5]],
        
        [[1, 0, 0,  3, 0],
         [3, 0, 4, -2, 2],
         [1, 0, 2,  0, 0],
         [3, 0, 1,  1, 0]]
    ])

    batch_multiplication_numpy = np.matmul(T1, T2)
    batch_multiplication_einsum = np.einsum("bij, bjk -> bik", T1, T2)

    # We can use einsum to sum up all the values in the tensor/array:
    sum_batch_multiplication_einsum = np.einsum("bij ->", batch_multiplication_einsum)

    # We can do row sums and column sums using einsum too:
    M3_row_sums = np.einsum("ij -> i", M3)
    M3_col_sums = np.einsum("ij -> j", M3)

    # Practical example - "Attention is all you need" paper:
    # For Q, K = batch size, K = batch size, and  s_q, s_k = model size
    Q = np.random.randn(32,  64, 512) # Dimensions bqm
    K = np.random.randn(32, 128, 512) # Dimensions bkm
    result = np.einsum("bqm, bkm -> bqk", Q, K)

    # Another practical example - "Reformer: The Efficient Transformer" paper:
    # Suppose A has shape (1, 4, 8) and B has shape (1, 4, 4). So, A has a batch
    # size of 1, sequence length of 4 and a model length of 8. In the paper, they
    # break the data up into "buckets", so A then has shape (1, 4, 4, 2) and B
    # has shape (1, 4, 4, 1). These buckets are arranged  by grouping columns.
    # Let A have indicies bcij and B have indicies bcik. We want to calculate the
    # result B.T A which has shape bckj which implies a shape of (1, 4, 1, 2).
    A = np.random.randn(1, 4, 4, 2)
    B = np.random.randn(2, 4, 4, 1)

    # We reversed bcik to bcki in B to get the transpose of the inner arrays! 
    result = np.einsum("bcki, bcij -> ", B, A) # Using einsum
    result = np.matmul(np.transpose(B, (0, 1, 3, 2)), A) # Using matmul


    # Common TensorFlow Functions:
    # ----------------------------

    # We can add an extra axis of length 1to an input tensor using tf.expand_dims()
    tensor_3d_to_4d = tf.expand_dims(input = tensor_3d, axis = 2, name  = None)

    # We can remove a dimension of length 1 of an input tensor with tf.squeeze()
    tensor_2d_to_1d = tf.squeeze(input = T4, axis = 1, name = None)

    # We can re-shape a tensor using tf.reshape(). Note that the elements have to
    # be able to fit in the new shape of the tensor! If one component of shape
    # is -1, the size of that dimension is made so the total size remains
    # constant. A shape of [-1] to any tensor flattens it completely. Only one
    # element of shape can be -1.
    tensor_2_4_5_to_1d = tf.reshape(tensor = T2, shape = [2*4*5], name = None)

    # We can concatenate tensors using tf.concat()
    concat_tensor_2d = tf.concat(values = [tensor_2d, M1], axis = 0, name = None)

    # Using tf.stack(), we can stack tensors along a new axis. Here, we stack
    # four tensors of shape (2, 3) to get a tensor of shape (4, 2, 3)
    stacked_2d_tensors = tf.stack(values = [M1, M1, M1, M1], axis = 0, name = None)
    
# ============================================================================ #
# Tensors and Variables - Code End                                             |
# ============================================================================ #
