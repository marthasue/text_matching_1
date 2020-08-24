import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from dssm.graph import Graph
import tensorflow as tf
from utils.load_data import load_char_data
from dssm import args
from tqdm import tqdm
import numpy as np
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)

p, h, y = load_char_data('input/train.csv',data_size=None)
p_eval, h_eval, y_eval = load_char_data('input/dev.csv',data_size=None)

p_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='p')
h_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='h')
y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')

dataset = tf.data.Dataset.from_tensor_slices((p_holder, h_holder, y_holder))
dataset = dataset.batch(args.batch_size).repeat(args.epochs)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

model = Graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

# with tf.name_scope("loss"):
#     loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=d2))
#     tf.summary.scalar("loss", loss)  # 记录loss
# with tf.name_scope("accuracy"):
#     correct_prediction = tf.equal(tf.argmax(d2, 1), y)
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.summary.scalar("acc", accuracy)  # 记录accuracy
# 模型保存器
saver = tf.train.Saver(max_to_keep=args.epochs)
# 合并记录操作
merge = tf.summary.merge_all()


with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={p_holder: p, h_holder: h, y_holder: y})
    steps = int(len(y) / args.batch_size)
    test_total_batch = int(len(y_eval) / args.batch_size) + 1
    # 日志记录
    train_writer = tf.summary.FileWriter(args.log_dir + "train/", sess.graph)  # 记录默认图
    test_writer = tf.summary.FileWriter(args.log_dir + "test/")
    for epoch in range(args.epochs):
        print('epoch: ' + str(epoch))
        # train process
        for i in tqdm(range(steps)):
            train_accuracy_list = []
            train_loss_list = []
            p_batch, h_batch, y_batch = sess.run(next_element)
            _, summary, train_loss, train_accuracy = sess.run([model.train_op,merge, model.loss, model.acc],
                                    feed_dict={model.p: p_batch,
                                               model.h: h_batch,
                                               model.y: y_batch,
                                               model.keep_prob: args.keep_prob})
            train_accuracy_list.append(train_accuracy)
            train_loss_list.append(train_loss)
        train_writer.add_summary(summary, epoch)
        print('train_acc:' + str(np.mean(train_accuracy_list)))
        print('train_loss:' + str(np.mean(train_loss_list)))
        # test process
        if (epoch + 1) % args.test_step == 0:
            test_accuracy_list = []
            test_loss_list = []
            for j in range(test_total_batch):
                summary, test_loss, test_accuracy = sess.run([merge, model.loss, model.acc],
                                                           feed_dict={model.p: p_eval,
                                                                      model.h: h_eval,
                                                                      model.y: y_eval,
                                                                      model.keep_prob: 1})
                test_accuracy_list.append(test_accuracy)
                test_loss_list.append(test_loss)
            test_writer.add_summary(summary, epoch)
            print('test_acc:' + str(np.mean(test_accuracy_list)))
            print('test_loss:' + str(np.mean(test_loss_list)))

    train_writer.close()
    test_writer.close()
    saver.save(sess, f'./model/dssm_{epoch}.ckpt')
    # for epoch in range(args.epochs):
    #     for step in range(steps):
    #         p_batch, h_batch, y_batch = sess.run(next_element)
    #         _, loss, acc = sess.run([model.train_op, model.loss, model.acc],
    #                                 feed_dict={model.p: p_batch,
    #                                            model.h: h_batch,
    #                                            model.y: y_batch,
    #                                            model.keep_prob: args.keep_prob})
    #         print('epoch:', epoch, ' step:', step, ' loss: ', loss, ' acc:', acc)
    #
    #     loss_eval, acc_eval = sess.run([model.loss, model.acc],
    #                                    feed_dict={model.p: p_eval,
    #                                               model.h: h_eval,
    #                                               model.y: y_eval,
    #                                               model.keep_prob: 1})
        #print('loss_eval: ', loss_eval, ' acc_eval:', acc_eval)
        #print('\n')
        #saver.save(sess, f'../output/dssm/dssm_{epoch}.ckpt')