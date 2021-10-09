# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import re
# # import urllib.request
# # from konlpy.tag import Okt
# # from sklearn.model_selection import train_test_split
# # from tensorflow.keras.preprocessing.text import Tokenizer
# # from tensorflow.keras.preprocessing.sequence import pad_sequences
# # from collections import Counter
# #
# # print(1)
# # urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename="ratings_total.txt")
# # total_data = pd.read_table('ratings_total.txt', names=['ratings', 'reviews'])
# # total_data['label'] = np.select([total_data.ratings > 3], [1], default=0)
# # total_data['ratings'].nunique(), total_data['reviews'].nunique(), total_data['label'].nunique()
# # total_data.drop_duplicates(subset=['reviews'], inplace=True)
# # print('총 샘플의 수 :',len(total_data))
# # train_data, test_data = train_test_split(total_data, test_size = 0.25, random_state = 42)
# # train_data['reviews'] = train_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# # train_data['reviews'].replace('', np.nan, inplace=True)
# # print(train_data.isnull().sum())
# # test_data.drop_duplicates(subset = ['reviews'], inplace=True) # 중복 제거
# # test_data['reviews'] = test_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
# # test_data['reviews'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
# # test_data = test_data.dropna(how='any') # Null 값 제거
# # print('전처리 후 테스트용 샘플의 개수 :',len(test_data))
# # okt = Okt()
# # print(okt.morphs('와 이런 것도 상품이라고 차라리 내가 만드는 게 나을 뻔', stem=True))
# # stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']
# # train_data['tokenized'] = train_data['reviews'].apply(okt.morphs)
# # train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
# # test_data['tokenized'] = test_data['reviews'].apply(okt.morphs)
# # test_data['tokenized'] = test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
# # negative_words = np.hstack(train_data[train_data.label == 0]['tokenized'].values) # 부정 단어
# # positive_words = np.hstack(train_data[train_data.label == 1]['tokenized'].values) # 긍정 단어
# # negative_word_count = Counter(negative_words)
# # print(negative_word_count.most_common(20)) # 부정 단어 count 출력
# # positive_word_count = Counter(positive_words)
# # print(positive_word_count.most_common(20)) # 긍정 단어 count 출력
# # X_train = train_data['tokenized'].values
# # y_train = train_data['label'].values
# # X_test= test_data['tokenized'].values
# # y_test = test_data['label'].values
# #
# # tokenizer = Tokenizer()
# # tokenizer.fit_on_texts(X_train)
# #
# # threshold = 2
# # total_cnt = len(tokenizer.word_index) # 단어의 수
# # rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
# # total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
# # rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합
# #
# # # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
# # for key, value in tokenizer.word_counts.items():
# #     total_freq = total_freq + value
# #
# #     # 단어의 등장 빈도수가 threshold보다 작으면
# #     if(value < threshold):
# #         rare_cnt = rare_cnt + 1
# #         rare_freq = rare_freq + value
# #
# # print('단어 집합(vocabulary)의 크기 :',total_cnt)
# # print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
# # print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
# # print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
# #
# # vocab_size = total_cnt - rare_cnt + 2
# # print('단어 집합의 크기 :',vocab_size)
# #
# # tokenizer = Tokenizer(vocab_size, oov_token = 'OOV')
# # tokenizer.fit_on_texts(X_train)
# # X_train = tokenizer.texts_to_sequences(X_train)
# # X_test = tokenizer.texts_to_sequences(X_test)
# #
# # y_train = np.array(train_data['label'])
# # y_test = np.array(test_data['label'])
# #
# # max_len = 50
# # X_train = pad_sequences(X_train, maxlen = max_len)
# # X_test = pad_sequences(X_test, maxlen = max_len)
# #
# #
# # from tensorflow.keras.layers import Embedding, Dense, LSTM
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# #
# # model = Sequential()
# # model.add(Embedding(vocab_size, 100))
# # model.add(LSTM(128))
# # model.add(Dense(1, activation='sigmoid'))
# #
# # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
# # mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
# #
# # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# # history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)
# #
# # loaded_model = load_model('best_model.h5')
# # print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
# #
# # def sentiment_predict(new_sentence):
# #   new_sentence = okt.morphs(new_sentence) # 토큰화
# #   new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
# #   encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
# #   print(encoded)
# #   pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
# #   print(pad_new)
# #   score = float(loaded_model.predict(pad_new)) # 예측
# #   if(score > 0.5):
# #     print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
# #   else:
# #     print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))
# #
# # sentiment_predict('맛있어요 정말')
# # print('success')
#
# vocab_size = 42019
# import csv
# import pandas as pd
# path = r'C:\Users\chaeyun\PycharmProjects\testtest\data\X_train.csv'
# # file = open(path)
# # X_train = csv.reader(file)
#
#
# X_train = []
# with open(path, 'r', encoding='UTF8') as f:
#     dr = csv.reader(f)
#     for list in dr:
#         X_train.append(list)
#
# print(s)