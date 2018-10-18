#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
import jieba
import word2vec
from tqdm import tqdm

def load_data_from_csv(filename,header = 0,encoding="utf-8"):
    df = pd.read_csv(filename,header=header,encoding=encoding)
    return df

def data_clean(dataframe,outpath,isTest = False):
    stop_words = ['.',',','···','、',' ','\n','\t','~']
    new_dataframe = pd.DataFrame(columns=dataframe.columns)

    for index,rows in tqdm(dataframe.iterrows()):
        sentence = rows["content"]
        new_sentence = ""
        for word in sentence:
            if word not in stop_words:
                new_sentence += word
        rows["content"] = new_sentence

        if not isTest:
            for columns,value in rows.iteritems():
                if columns != "content":
                    if np.isnan(float(rows[columns])) == True:
                        rows[columns] = -2
                        print("遇到空值：{}".format(index))
        new_dataframe.loc[index] = rows

    print("清洗完成")

    new_dataframe.to_csv(outpath,index = None)
    return new_dataframe


def seg_words(contents):
    contents_segs = []
    for content in contents:
        segs = jieba.cut(content)
        contents_segs.append(" ".join(segs))

    return contents_segs


train_file = "./data/trainingset/sentiment_analysis_trainingset.csv"
validation_file = "./data/validationset/sentiment_analysis_validationset.csv"
test_file = "./data/testa/sentiment_analysis_testa.csv"
train_after_clean = "./data/train_after_clean.csv"
val_after_clean = "./data/val_after_clean.csv"
test_after_clean = "./data/test_after_clean.csv"
seg_text = "./data/seg_list.txt"
embedding_bin = "./embedding.bin"



def preprocess():
    train_df = load_data_from_csv(train_file)
    val_df = load_data_from_csv(validation_file)
    test_df = load_data_from_csv(test_file)


    train_df = data_clean(train_df,train_after_clean)
    val_df = data_clean(val_df,val_after_clean)
    test_df = data_clean(test_df,test_after_clean)


    train_content = train_df.iloc[:,1]
    val_content = val_df.iloc[:,1]
    test_content = test_df.iloc[:,1]

    all_content = []
    all_content.extend(train_content)
    all_content.extend(val_content)
    all_content.extend(test_content)

    print(len(all_content))

    all_seg_words = seg_words(all_content)

    with open(seg_text,"w+") as txt_write:
        for sentence in tqdm(all_seg_words):
            sentence = sentence.replace("\n","") + "\n"

            txt_write.write(sentence)
    txt_write.close()

    word2vec.word2vec(seg_text,embedding_bin,min_count = 5,size = 100,verbose = True)

preprocess()





















