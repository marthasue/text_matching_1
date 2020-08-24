import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from dssm.graph import Graph
from dssm.graph import TestGraph
import tensorflow as tf
from utils.load_data import load_char_data,char_index
import pandas as pd
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

p, h, y = load_char_data('input/test.csv', data_size=None)


if __name__ == '__main__':

    # print(h.shape, p.shape)
    model = TestGraph()
    saver = tf.train.Saver()
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './model/dssm_49.ckpt')
        acc = sess.run(model.acc,
                             feed_dict={model.p: p,
                                        model.h: h,
                                        model.y: y,
                                        model.keep_prob: 1})
    print(acc)
    # correct_count = 0
    # for i in correct_prediction:
    #     if i==True:
    #         correct_count = correct_count+1
    # print(correct_count)
        # print(np.sum(correct_prediction))
    # path = os.path.join(os.path.dirname(__file__), '../' + '/input/test.csv')
    # df = pd.read_csv(path,sep='\t')
    # label = df.loc[:,'label']
    # df2 = df
    # correct_count = 0
    # for i in df.index:
    #     #df2.loc[i,'label'] = res[i]
    #     if correct_prediction[i]==label[i]:
    #         correct_count=correct_count+1
    # print(correct_count)
    #df2.to_excel('predict_data.xlsx','Sheet1')
