#!/usr/bin/env python3
"""task 3"""
import numpy as np
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """function that trains a loaded neural network
    model using mini-batch gradient descent"""
    m = X_train.shape[0]
    nsaver = tf.train.import_meta_graph(load_path + '.meta')
    with tf.Session() as sess:
        nsaver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]

        for epo in range(epochs):
            t_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            t_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            v_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            v_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            X_train, Y_train = shuffle_data(X_train, Y_train)
            print("After {} epochs:".format(epo))
            print("\tTraining Cost: {}".format(t_cost))
            print("\tTraining Accuracy: {}".format(t_acc))
            print("\tValidation Cost: {}".format(v_cost))
            print("\tValidation Accuracy: {}".format(v_acc))
            for batch in range(m // batch_size):
                bs = batch_size
                idx = batch * batch_size
                if (idx + 1) * batch_size > m:
                    bs = m - idx
                X_batch = X_train[idx: bs + idx]
                Y_batch = Y_train[idx: bs + idx]
                feed_dict = {x: X_batch, y: Y_batch}
                if batch != 0 and batch % 100 == 0:
                    step_cost = sess.run(loss, feed_dict=feed_dict)
                    step_accuracy = sess.run(accuracy, feed_dict=feed_dict)
                    print("\tStep {}:".format(batch))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
        return nsaver.save(sess, save_path)
