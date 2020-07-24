import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt

# 启用动态图机制
tf.enable_eager_execution()
#调用input_data.py中的read_data_sets函数获取数据
mnist_data = input_data.read_data_sets('mnist_data/', one_hot=False)

train_images = mnist_data.train.images
train_labels = mnist_data.train.labels

test_images = mnist_data.test.images
test_labels = mnist_data.test.labels

#tf.keras.Input函数用于向模型中输入数据，并指定数据的形状、数据类型等信息。
input_ = tf.keras.Input(shape=(784, ))

fc1 = tf.keras.layers.Dense(128, activation='tanh')(input_)
fc2 = tf.keras.layers.Dense(32, activation='tanh')(fc1)
out = tf.keras.layers.Dense(1)(fc2)

# 使用inputs与outputs建立函数链式模型；
model = tf.keras.Model(inputs=input_, outputs=out)

#使用keras构建深度学习模型，我们会通过model.summary()输出模型各层的参数状况
model.summary()

#构建模型后，通过调用compile方法配置其训练过程：
model.compile(loss='mse',optimizer='adam')#mean_squared_error=mse
                        # 顾名思义，意为均方误差，也称标准差，缩写为MSE，可以反映一个数据集的离散程度。
                        #标准误差定义为各测量值误差的平方和的平均值的平方根，故又称为均方误差。
                        # model.compile (optimizer=Adam(lr=1e-4), loss=’binary_crossentropy’, metrics=[‘accuracy’])

#模型拟合
model.fit(x=train_images, y=train_labels, epochs=5)

for i in range(10):
    #tf.expand_dims用来增加维度，
    pred = model(tf.expand_dims(test_images[i], axis=0))
    img = np.reshape(test_images[i], (28, 28))
    lab = test_labels[i]
    print('真实标签: ', lab, '， 网络预测: ', pred.numpy())
    plt.imshow(img)
    plt.show()


#
# import tensorflow.examples.tutorials.mnist.input_data as input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# x = tf.placeholder(tf.float32, [None, 784])
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10])+0.1)
# y = tf.nn.softmax(tf.matmul(x,W) + b)
#
# y_ = tf.placeholder("float", [None,10])
#
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#
# init = tf.global_variables_initializer()
#
# sess = tf.Session()
# sess.run(init)
#
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
