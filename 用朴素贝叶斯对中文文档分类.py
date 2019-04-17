# -*- coding:utf-8 -*-
import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#通过训练已分好类的的文档内容和label,拿测试集的数据去验证
#加载数据后进行分词处理，过滤掉停用词及tfidf低的词，做fit和predict
#1.加载数据
# 加载停用词表
file = './text_classification-master/text classification/stop/stopword.txt'
stopWords = [line.strip() for line in open(file, 'r', encoding='utf-8').readlines()]

labelMap = {'体育':0,'女性':1,'文学':2,'校园':3}
#2.加载训练数据和测试数据
def loadData(basePath):
    '''
    :param basePath:基础路径
    :return:分词列表，标签列表
    '''
    documents = []
    labels = []
    for root, dirs ,files in os.walk(basePath):
        for file in files:
            #print(root)
            label = root.split('\\')[-1] #windows上路径符号自动转成\了，所以要转义
            labels.append(label)
            #print(label)
            filename = os.path.join(root, file) #每个文件的名称./text_classification-master/text classification/test\体育\1451.txt
            #print(filename)
            with open(filename, 'rb') as f:   #因为字符集问题因此直接用二进制方式读取
                content = f.read()
                word_list = list(jieba.cut(content))
                words = [wl for wl in word_list]
                documents.append(''.join(words))
    # print(labels)
    # print(len(labels))
    return documents, labels


trainDocuments, trainLabels =  loadData('./text_classification-master/text classification/train')
testDocuments, testLabels = loadData('./text_classification-master/text classification/test')

#3分词方式计算训练集单词权重（计算单词权重，其实是为了过滤掉一些停用词，增加计算速率）
tfidf_vec = TfidfVectorizer(tokenizer=jieba.cut, stop_words=stopWords, max_df=0.5)
#max_df描述单词在文档中的出现率，如果max_df=0.5,则在文档中出现概率50%，包含的信息少，不作为分词统计
train_features = tfidf_vec.fit_transform(trainDocuments)
print(train_features.shape)

#4.生成朴素贝叶斯模型
clf = MultinomialNB(alpha=0.001).fit(train_features, trainLabels)

#5.使用生成的分类器做预测
#分词方式计算测试集单词权重
test_tf = TfidfVectorizer(tokenizer=jieba.cut, stop_words=stopWords, max_df=0.5, vocabulary=tfidf_vec.vocabulary_)
#tfidf_vec.vocabulary_ 是每个单词的ID
test_features = test_tf.fit_transform(testDocuments)
predicted_labels = clf.predict(test_features)

#6.计算准确率
print(metrics.accuracy_score(testLabels, predicted_labels))

