# import tokenizer as tokenizer
import pandas as pd
from django.shortcuts import render

# from real_test import sentiment_predict
from testapp.models import Content
from konlpy.tag import Okt
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
#
okt = Okt()
model = load_model('models/review_model.h5')

vocab_size = 42019
X_train = []
path = r'C:\Users\chaeyun\PycharmProjects\testtest\data\X_train.csv'
with open(path, 'r', encoding='UTF8') as f:
    reader = csv.reader(f)
    for idx, list in enumerate(reader):
        X_train.append(list)


# X_train = pd.read_csv('data/X_train.csv')
# X_train = X_train[1:]
# # y_train = pd.read_csv('data/y_train.csv')
# y_train = y_train['a']
# X_test = pd.read_csv('data/X_test.csv')
# X_test = X_test['a']
# y_test = pd.read_csv('data/y_test.csv')
# y_test = y_test['a']
#
max_len = 50
tokenizer = Tokenizer(vocab_size, oov_token = 'OOV')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=max_len)
# X_test = tokenizer.texts_to_sequences(X_test)

stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']


def index(request):
    return render(request, 'testapp/index.html')

def result(request):
    full = request.GET['fulltext']
    full = okt.morphs(full)
    full = [word for word in full if not word in stopwords]
    encoded = tokenizer.texts_to_sequences([full])
    # error !!!!
    pad_new = pad_sequences(encoded, maxlen=80)
    score = float(model.predict(pad_new))
    if score > 0.5:
        output = '긍정'
    else:
        output = '부정'
    return render(request, 'testapp/result.html', {'output': output})
    # words = full.split()
    # words_dic = {}
    # for word in words:
    #     if word in words_dic:
    #         words_dic[word] += 1
    #     else:
    #         words_dic[word] = 1
    # return render(request, 'testapp/result.html',
    #               {'full':full, 'length':len(words), 'dic':words_dic.items()})