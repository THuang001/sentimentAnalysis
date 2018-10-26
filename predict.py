# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import pandas as pd
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='4'

test_after_clean = pd.read_csv(r"./data/test_after_clean.csv")
embedding_bin = "./embedding.bin"

ckpt_path = ""

datasets = []
ids = []
for index, rows in test_after_clean.iterrows():
    this_id = rows["id"]
    ids.append(this_id)
    x = list(jieba.cut(rows["content"]))
    if len(x) > content_limit:
        x = x[:content_limit]
    datasets.append({"id": this_id, "content": x})

word2vec_model = word2vec.load(embedding_bin)

index = 0
pre_input = np.zeros(shape=[len(datasets), content_limit, 100], dtype=np.float32)

for one in datasets:
    for i in range(len(one["content"])):
        if one["content"][i] in word2vec_model:
            pre_input[index, i, :] = word2vec_model[one["content"][i]]
    index += 1

saver = tf.train.import_meta_graph(ckpt_path + '/model_name.meta')

with tf.Session(config= tf.ConfigProto(log_device_placement=True)) as sess:
    saver.restore(sess, ckpt_path)
    sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))

    val_logits = sess.run(logits, feed_dict={inputs: pre_input})
    val_prediction = predict(val_logits)

for i in range(2, len(datasets) + 2):
    test_after_clean.iloc[[i], 2:23] = np.array(val_prediction.iloc[[i]])
    test_after_clean.to_csv("./data/predict.csv")











