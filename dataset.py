import random
import jieba
import word2vec
import numpy as np
from data_preprocess import load_data_from_csv,train_after_clean,val_after_clean,test_after_clean
from data_preprocess import seg_words


content_limit = 500

def get_passage_limit():
    train_df = load_data_from_csv(train_after_clean)
    val_df = load_data_from_csv(val_after_clean)
    test_df = load_data_from_csv(test_after_clean)

    train_content = train_df.iloc[:,1]
    val_content = val_df.iloc[:,1]
    test_content = test_df.iloc[:,1]

    all_content = []
    all_content.extend(train_content)
    all_content.extend(val_content)
    all_content.extend(test_content)

    print(len(all_content))

    all_seg_words = seg_words(all_content)
    max = 0
    max_len = []
    for sentence in all_seg_words:
        sentence = sentence.replace("\n"," ") + "\n"
        if len(sentence) > max :
            max = len(sentence)
            max_len = sentence

    return max,max_len



def get_dataset(file):
    dataframe = load_data_from_csv(file)

    dataset = []
    for index,rows in dataframe.iterrows():
        ids = rows["id"]
        x = list(jieba.cut(rows["content"]))
        if len(x) > content_limit:
            x = x[:content_limit]

        y = list(rows[2:])

        dataset.append({"id":ids,"content":x,"label":y})

    return dataset

def batch_generator(dataset,batch_size,word2vec_bin,shuffle = True):

    word2vec_model = word2vec.load(word2vec_bin)

    if shuffle:
        random.shuffle(dataset)

    data_num = len(dataset)
    batch_count = 0

    while True:
        if batch_count * batch_size + batch_size > data_num:
            one_batch = dataset[batch_count * batch_size:data_num]
            for i in range(batch_count * batch_size + batch_size - data_num):
                one_batch.append[dataset[i]]
            batch_count = 0
            if shuffle:
                random.shuffle(dataset)
        else:
            one_batch = dataset[batch_count * batch_size:batch_count * batch_size + batch_size]
            batch_count += 1


        index = 0
        one_batch_ids = []
        one_batch_inputs = np.zeros(shape=[batch_size,content_limit,100],dtype=np.float32)
        one_batch_labels = []

        for one in one_batch:
            one_batch_ids.append(one["id"])
            one_batch_labels.append(one["label"])
            for i in range(len(one("content"))):
                if one("content")[i] in word2vec_model:
                    one_batch_inputs[index,i,:] = word2vec_model[one["content"][i]]

            index += 1

        one_batch_ids = np.array(one_batch_ids)
        one_batch_labels = np.array(one_batch_labels)

        yield one_batch_ids,one_batch_inputs,one_batch_labels






























