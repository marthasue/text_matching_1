import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from dssm.graph import TestGraph
import tensorflow as tf
from utils.load_data import char_index_single, char_index
import pandas as pd
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_char_data_for(file, data_size=None):
    path = os.path.join(os.path.dirname(__file__), '../' + file)
    df = pd.read_csv(path,sep = '\t')
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    # [1,2,3,4,5] [4,1,5,2,0]
    _, h_c_index = char_index(p, h)

    return h, h_c_index, label

if __name__ == '__main__':
    sentence_h, h, y = load_char_data_for('input/test.csv', data_size=None)
    source_sentence = '主办方诸暨一百集团有限公司的主营产品是液体乳,咖啡饮料,其他水产加工品,糖果,即食方便食品,其他熟肉制品,其他水果、坚果加工品,瓶装葡萄酒，容器瓶装≤2升,其他食品添加剂，所在行业是农业,食品制造业,酒、饮料和精制茶制造业，目标产品是各种消费品批发，目标行业是其他批发业'
    p = char_index_single(source_sentence, 200)
    p = np.repeat(p, h.shape[0], axis=0)
    # print(h.shape, p.shape)
    model = TestGraph()
    saver = tf.train.Saver()
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './model/dssm_49.ckpt')
        res = sess.run(model.prediction,
                             feed_dict={model.p: p,
                                        model.h: h,
                                        model.y: y,
                                        model.keep_prob: 1})
        sentence_sim = dict()
        for sentence, x in zip(sentence_h, res):
            sentence_sim[sentence] = x
        rank = sorted(sentence_sim.items(), key=lambda e: e[1], reverse=True)
        print("source: %s" % source_sentence)
        print("target: \n")
        for i in rank[:30]:
            print("%s\t%s" % (np.round(i[1], 6), i[0]))

