# coding=UTF-8
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

#保存为pb模型
def export_model(session, m):
   #只需要修改这一段，定义输入输出，其他保持默认即可
    model_signature = signature_def_utils.build_signature_def(
        inputs={"input": utils.build_tensor_info(m.p)},
        outputs={
            "output": utils.build_tensor_info(m.y)},

        method_name=signature_constants.PREDICT_METHOD_NAME)

    export_path = "pb_model"
    # if os.path.exists(export_path):
    #     os.system("rm -rf "+ export_path)
    print("Export the model to {}".format(export_path))

    try:
        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')
        builder = saved_model_builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            session, [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    model_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()
    except Exception as e:
        print("Fail to export saved model, exception: {}".format(e))



if __name__ == "__main__":

    graph2 = tf.Graph()
    with graph2.as_default():
        m = Graph()
        saver = tf.train.Saver()
    with tf.Session(graph=graph2) as session:
        saver.restore(session, "model/dssm_9.ckpt") #加载ckpt模型
        export_model(session, m)

    #load_pb()
