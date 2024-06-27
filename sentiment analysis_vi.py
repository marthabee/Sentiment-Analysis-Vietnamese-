from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from pyvi import ViTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential 
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fb = pd.read_csv('fb_vn.csv')
#df = pd.read_csv('D:\\OneDrive - CMC University\\CMCU-2024\DACN 2024\\Facebook Sentiment Analysis\\Sentiment-Analysis\\fb_vn.csv')


fb.columns = map(str.lower, fb.columns)
fb.shape

fb['label'] = fb['label'].apply(lambda x: str(int(x)))

max_features = 2000

fb['message'] = fb['message'].apply(ViTokenizer.tokenize)


tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(fb['message'].values)
X = tokenizer.texts_to_sequences(fb['message'].values)
X = pad_sequences(X)

print(fb.head(10))


fb.label.value_counts()

Y = pd.get_dummies(fb['label']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


embed_dim = 200
lstm_out = 200

model = Sequential()
#model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(Embedding(max_features, embed_dim))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


batch_size = 32
hist = model.fit(X_train, Y_train, epochs = 30, batch_size=batch_size, verbose = 2)


history = pd.DataFrame(hist.history)
plt.figure(figsize=(7,7))
plt.plot(history["loss"])
plt.plot(history["accuracy"])
plt.title("Loss and accuracy of model")
plt.show()

#Testing the model, and retrieveing score and accuracy:
score,acc = model.evaluate(X_test,Y_test)
print("score: %.2f" % (score))
print("accuracy: %.2f" % (acc))


validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
x_test = X_test[:-validation_size]
y_test = Y_test[:-validation_size]


pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):
    result = model.predict(X_validate[x].reshape(1,x_test.shape[1]),verbose = 2)[0]
    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1
print("positive_acc", pos_correct/pos_cnt*100, "%")
print("negative_acc", neg_correct/neg_cnt*100, "%")


cmnt = ['trung bình thôi']
cmnt = tokenizer.texts_to_sequences(cmnt)
cmnt = pad_sequences(cmnt, maxlen=203, dtype='int32', value=0)
print(cmnt)
sentiment = model.predict(cmnt,batch_size=2,verbose = 2)[0]
print(sentiment)
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("neutral")
else:
    print("positive")