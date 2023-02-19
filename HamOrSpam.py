import pandas as pd #read csv
import numpy as np 
import matplotlib.pyplot as plt #just plot
import seaborn as sns
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Dense, Activation, Input, Embedding, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical, pad_sequences
from keras.callbacks import EarlyStopping
df=pd.read_csv('spam.csv')
df.head(5)
df.Category.value_counts().plot.bar()
X=df.Message
Y=df.Category
le=LabelEncoder() #call LabelEncoder and name it le
Y=le.fit_transform(Y) #use LabelEnocder to map labels to integer values
print(X)
print(Y)
Y=Y.reshape(-1,1) 
print(Y)
print(Y.shape) 
trainx, testx, trainy, testy= train_test_split(X, Y, test_size=0.15)
#trainx-traning data for message 
#trainy-training data for category
print(trainx)
print(len(trainx)) #training length-4736 (approx 85%)  testing length-836 (15%)
max_words=1000 #will print 1000 most frequent vocab
max_len=150 #our preferred size of vector/word
tk=Tokenizer(num_words=max_words) #called tokenizer and named it tk
tk.fit_on_texts(trainx) #trainx cuz its the message that needs to be broken down into words so it can be embedded
words=tk.texts_to_sequences(trainx)
print(words)
print(len(words))
#to make all vectors of same size -padding
#to make all vectors of same size -padding
pwords=keras.utils.pad_sequences(words, maxlen=max_len)
print(pwords) 
print(len(pwords[0])) #each word is of length 150 now
#creating model using functional API
inputs=Input(shape=[max_len]) #input layer(the input shape=max_len=150 cuz length of each word is 150)
layer=Embedding(max_words, 64, input_length=max_len)(inputs)#Embedding Layer
layer=LSTM(64)(layer)
#layer=Dense(256)(layer)
#layer=Activation('relu')(layer)
#layer=Dropout(0.5)(layer)
layer=Dense(1)(layer)
layer=Activation('sigmoid')(layer)
model=Model(inputs=inputs, outputs=layer)
model.summary()
model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
#fitting our training data on the model created
history=model.fit(pwords, trainy, batch_size=128, epochs=10, validation_split=0.2) 
test1=["Please send your bank account details to 8235860944. There is an urgent message waiting for you"]
test2=["IRS is filling a lawsuit against you, call on 980737494 URGENTLY"]
test3=["Dear Customer, Bank of India has closed your account. Please confirm your pin at www.bankofindia.com to keep your account activated"]
test=tk.texts_to_sequences(test2)
test=keras.utils.pad_sequences(test, maxlen=max_len)
ans=model.predict(test)
print(ans)
if(ans<0.5):
    print("HAM") #from above we know 0 is ham and 1 is spam
else:
    print("SPAM")
plt.plot(history.history['accuracy'], 'r', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'g', label='Validation Accuracy')
plt.title('Training VS Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(history.history['loss'], 'r', label='Training Loss')
plt.plot(history.history['val_loss'], 'g', label='Validation Loss')
plt.title('Training VS Validation Loss')
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
