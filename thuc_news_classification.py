# -*- coding: utf-8 -*-
import pandas as pd
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from jieba.analyse import extract_tags  # tfidf的效果优于textrank
import jieba


def file_filter(file):
    """
    :param file: 内容为中文，需要将文章分为空格分隔利用jieba分词
    :return: 利用tf-idf值过滤非重要字后的词语组成的file
    """
    # 分词得到列表
    word_list = jieba.lcut_for_search(file)
    # 得到前20个关键字
    words = extract_tags(file, topK=20, withWeight=False)
    temp = [word for word in word_list if word in words]
    new_content = ' '.join(temp)
    return new_content


def get_news_data():
    """
    不同类别文件夹，表示该文本类别，文本数据过多，取每类的10000条作为训练集
    :return: 包含关键信息的文本数据，与label的列表
    """
    label_dirs = os.listdir('../thuc_news')
    total_files = list()
    for label_dir in label_dirs:
        # 每个dir一个类别的文件
        path = '../thuc_news/' + label_dir
        files = os.listdir(path)
        # 每类随机选择1000个样本
        random_files = random.sample(files, 1000)
        for file in random_files:
            with open('../thuc_news/' + label_dir + '/' +file, encoding='utf-8') as f:
                content = f.read()
                # 将文本内容过滤
                new_content = file_filter(content)
                total_files.append({'cont': new_content, 'label': label_dir})
    return total_files


def classification():
    """
    体育  娱乐  家居  彩票  房产  教育  时尚  时政  星座  游戏  社会  科技  股票  财经
    :return: 
    """
    # 贝叶斯label可以识别中文
    # temp = ['体育', '娱乐', '家居', '彩票', '房产', '教育', '时尚', '时政', '星座', '游戏', '社会', '科技', '股票', '财经']
    # temp_dict = dict(enumerate(temp))
    # class_num = {v:k for k,v in temp_dict.items()}
    total_data = get_news_data()
    # 拆分训练集，测试集
    total = pd.DataFrame(total_data)
    # print(total.label.values)
    # print([class_num[i] for i in total.label.values])
    # total.label = [class_num[i] for i in total.label.values]  # 将label编码
    x_train, x_test, y_train, y_test = train_test_split(total.cont, total.label, test_size=0.75)
    # 构造训练集
    tfidf = TfidfVectorizer()
    values = tfidf.fit_transform(x_train.to_list()).toarray()
    # 953列，即fit进入953个词
    train_features = pd.DataFrame(values, columns=tfidf.get_feature_names())
    # 准备label
    # train_label = pd.get_dummies(y_train)  one-hot编码
    # print(train_label)
    nb = MultinomialNB(alpha=1.0)
    nb.fit(train_features, y_train)
    # 预测
    test_values = tfidf.transform(x_test).toarray()  # 转换后为parse矩阵需要toarray转为array
    test_features = pd.DataFrame(test_values, columns = tfidf.get_feature_names())
    # print(nb.predict_proba(test_features.loc[[0],:]))
    y_predict = nb.predict(test_features)
    print(nb.score(test_features, y_test))
    print(classification_report(y_test, y_predict))

if __name__ == '__main__':
    classification()
