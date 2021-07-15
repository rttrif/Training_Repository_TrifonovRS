# Раздел 2: Deep Learning and TensorFlow Fundamentals
# ===================================================

# 1. Create a vector, scalar, matrix and tensor with values of your choosing using tf.constant().
# 2. Find the shape, rank and size of the tensors you created in 1.
# 3. Create two tensors containing random values between 0 and 1 with shape [5, 300].
# 4. Multiply the two tensors you created in 3 using matrix multiplication.
# 5. Multiply the two tensors you created in 3 using dot product.
# 6. Create a tensor with random values between 0 and 1 with shape [224, 224, 3].
# 7. Find the min and max values of the tensor you created in 6 along the first axis.
# 8. Created a tensor with random values of shape [1, 224, 224, 3] then squeeze it to change the shape to [224, 224, 3].
# 9. Create a tensor with shape [10] using your own choice of values, then find the index which has the maximum value.
# 10. One-hot encode the tensor you created in 9.

# ===================================================
# %%
import tensorflow as tf

print(tf.__version__)
# %%
# 1. Create a vector, scalar, matrix and tensor with values of your choosing using tf.constant().

# Scalar
scalar = tf.constant(9)
print(scalar)

# Vector
vector = tf.constant([14, 37, 49])
print(vector)

# Matrix
matrix = tf.constant([[10, 11, 56],
                      [16, 15, 19],
                      [66, 77, 88]])
print(matrix)

# Tensor
tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [10, 11, 12]],
                      [[13, 14, 15],
                       [16, 17, 18]]])
print(tensor)
# %%
# 2. Find the shape, rank and size of the tensors you created in 1.

# Shape
print(scalar.shape)
print(vector.shape)
print(matrix.shape)
print(tensor.shape)
#
# # Rank: The number of tensor dimensions.
print(scalar.ndim)
print(vector.ndim)
print(matrix.ndim)
print(tensor.ndim)

# Size
print(tf.size(scalar))
print(tf.size(vector))
print(tf.size(matrix))
print(tf.size(tensor))
# %%
# 3. Create two tensors containing random values between 0 and 1 with shape [5, 300].
random_tensor_1 = tf.random.uniform(shape=(5, 300))
print(random_tensor_1, random_tensor_1.shape)

random_tensor_2 = tf.random.uniform(shape=(5, 300))
print(random_tensor_2, random_tensor_2.shape)
# %%
# 4. Multiply the two tensors you created in 3 using matrix multiplication.
random_tensor_3 = random_tensor_1 * random_tensor_2
print(random_tensor_3)
# %%
# 5. Multiply the two tensors you created in 3 using dot product.
random_tensor_4 = random_tensor_1 @ tf.reshape(random_tensor_2, shape=(300, 5))
print(random_tensor_4)
# %%
# 6. Create a tensor with random values between 0 and 1 with shape [224, 224, 3].
random_tensor_5 = tf.random.uniform(shape=(224, 224, 3))
print(random_tensor_5)
# %%
# 7. Find the min and max values of the tensor you created in 6 along the first axis.

# Maximum
print(tf.reduce_max(random_tensor_5))
# Minimum
print(tf.reduce_min(random_tensor_5))

# %%
# 8. Created a tensor with random values of shape [1, 224, 224, 3] then squeeze it to change the shape to [224, 224, 3].
random_tensor_6 = tf.random.normal(shape=(1, 224, 224, 3))
print(random_tensor_6)

random_tensor_7 = tf.squeeze(random_tensor_6)
print(random_tensor_7)
# %%
# 9. Create a tensor with shape [10] using your own choice of values, then find the index which has the maximum value.
random_tensor_8 = tf.random.normal(shape=(10,))
print(random_tensor_8)

print(tf.argmax(random_tensor_8))
# %%
# 10. One-hot encode the tensor you created in 9.
some_list = [0, 1, 2, 3, 4, 6, 7, 8, 8, 10]

# One hot encode them
print(tf.one_hot(some_list, depth=8))
