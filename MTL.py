#import sys
#reload(sys)
#sys.setdefaultencoding("ISO-8859-1")
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#from keras.backend.tensorflow_backend import set_session
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.log_device_placement = True
#set_session(tf.Session(config=config))
import tensorflow as tf
from keras.engine import Layer
import keras.backend as K
from keras.layers import Layer
from keras.utils import CustomObjectScope
from keras import backend as K
import numpy as np
import gensim
import math
import codecs
import pandas as pd
from numpy import zeros
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from keras.models import Sequential
from keras.layers import Input,Dense,Embedding,Flatten,Conv1D,GlobalMaxPooling1D,Dropout,MaxPooling1D,Lambda
from keras.layers import Activation,Bidirectional,GRU,LSTM,Flatten
from keras.models import Model
from keras import optimizers
from keras.utils import np_utils
from keras.layers.merge import concatenate
from numpy import argmax
#from itertools import izip
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from keras.utils.vis_utils import plot_model
from sklearn.metrics import f1_score
word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary = True)
from sklearn.model_selection import StratifiedKFold
from math import log
import fastText as ft
model_ft = ft.load_model("wiki.en.bin")



Embedding_Dim = 300
Max_length = 40
Num_Words = 60000
seed = 7
np.random.seed(seed)



#Loading Embedding for fasttext

embeddings_index_fasttext = {}
g = codecs.open('cc.en.300.vec')
for line in g:
	values = line.rstrip().rsplit(' ')
	word = values[0]
	coefs = np.asarray(values[1:],dtype = 'float32')
embeddings_index_fasttext[word] = coefs
g.close()







counter = 0                                      #Number of unique tokens in a merge set of all data
Full_Dict = {}                              
with open("D1.txt","r") as a: 
	for line in a:
		#print line
		line = line.lower()
		line = line.split()
		for item in line:
			#item = item.strip()
			if item not in Full_Dict:
				Full_Dict[item] = counter
				counter += 1	
voc_1 = len(Full_Dict)     
#print ("The length of unique element in First Data is",voc_1)


#Tokenize and padding the sequences for Data 1
with open("D1.txt","r")as a:
	texts_1 = a.readlines()
#print texts_1
tokenizer_1 = Tokenizer(Num_Words)
tokenizer_1.fit_on_texts(texts_1)
Task_1 =  tokenizer_1.texts_to_sequences(texts_1)
data1 = pad_sequences(Task_1, maxlen=Max_length,padding = 'post')
word_index_1 = tokenizer_1.word_index   #dictionary of unique words of data 1
invert_1 = dict(map(reversed, word_index_1.items()))
vocab_size_1 = min(len(word_index_1)+1,(Num_Words))  #Number of unique tokens

#Preparing google word2vec embedding matrix for Data 1
embeddings_matrix_1 = np.zeros((vocab_size_1, Embedding_Dim))
for word, i in word_index_1.items():
	if i>= Num_Words:
		continue
	try:
		embedding_vector = word_vectors[word]
		embeddings_matrix_1[i] = embedding_vector
	except KeyError:
		embeddings_matrix_1[i] = np.random.normal(0, np.sqrt(0.25), Embedding_Dim)

#PREPARING Fasttext embedding for Data 1.
Embedding_Matrix_Fasttext_1 = np.zeros((vocab_size_1, Embedding_Dim))
#print "embedding_matrix_fasttext shape", Embedding_Matrix_Fasttext_1.shape
for word, i in word_index_1.items():
	#print i,word
    	if i > Num_Words:
        	continue
	embedding_vector = embeddings_index_fasttext.get(word)
	if (embedding_vector is not None) and len(embedding_vector) > 0:
		Embedding_Matrix_Fasttext_1[i] = embedding_vector
		#words_found.append(word)
	else:
		#words_not_found.append(word)
		Embedding_Matrix_Fasttext_1[i] = model_ft.get_word_vector(word)
#print "The shape of fast text is",Embedding_Matrix_Fasttext_1.shape


#concatenation of word2vec and fasttext embedding matrix for Data 1
concatenated_embedding_matrix_1 = np.hstack((embeddings_matrix_1,Embedding_Matrix_Fasttext_1))



flag = counter   #The count of unique tokens is assigned to "flag" variable
with open("D2.txt","r") as b:
	for line in b:
		line = line.lower()
		line = line.split()
		for item in line:
			#item = item.strip()
			if item not in Full_Dict:
				Full_Dict[item] = flag
				flag +=1
				

#Tokenize and padding the sequences for Data 2
with open("D2.txt","r")as b:
	texts_2 = b.readlines()
tokenizer_2 = Tokenizer(Num_Words)
tokenizer_2.fit_on_texts(texts_2)
Task_2 =  tokenizer_2.texts_to_sequences(texts_2)
data2 = pad_sequences(Task_2, maxlen=Max_length,padding = 'post')
word_index_2 = tokenizer_2.word_index     #dictionary of unique words of data 2
invert_2 = dict(map(reversed, word_index_2.items()))
vocab_size_2 = min(len(word_index_2)+1,(Num_Words))   #Number of unique tokens of data 2

#Preparing google word2vec embedding matrix for Data 2
embeddings_matrix_2 = np.zeros((vocab_size_2, Embedding_Dim))
for word, i in word_index_2.items():
	if i>= Num_Words:
		continue
	try:
		embedding_vector = word_vectors[word]
		embeddings_matrix_2[i] = embedding_vector
	except KeyError:
		embeddings_matrix_2[i] = np.random.normal(0, np.sqrt(0.25), Embedding_Dim)
		

#PREPARING Fasttext embedding for Data 2.
Embedding_Matrix_Fasttext_2 = np.zeros((vocab_size_2, Embedding_Dim))
for word, i in word_index_2.items():
    	if i > Num_Words:
        	continue
	embedding_vector = embeddings_index_fasttext.get(word)
	if (embedding_vector is not None) and len(embedding_vector) > 0:
		Embedding_Matrix_Fasttext_2[i] = embedding_vector
	else:
		Embedding_Matrix_Fasttext_2[i] = model_ft.get_word_vector(word)

#concatenation of word2vec and fasttext embedding matrix for Data 2
concatenated_embedding_matrix_2 = np.hstack((embeddings_matrix_2,Embedding_Matrix_Fasttext_2))



#The count of unique tokens is assigned to "Count" variable
Count = flag
with open("D3.txt","r") as c:
	for line in c:
		line = line.lower()
		line = line.split()
		for item in line:
			if item not in Full_Dict:
				Full_Dict[item] = Count
				Count +=1
#Tokenize and padding the sequences for Data 3
with open("D3.txt","r")as c:
	texts_3 = c.readlines()
tokenizer_3 = Tokenizer(Num_Words)
tokenizer_3.fit_on_texts(texts_3)
Task_3 =  tokenizer_3.texts_to_sequences(texts_3)
data3 = pad_sequences(Task_3, maxlen=Max_length,padding = 'post')
word_index_3 = tokenizer_3.word_index                   #dictionary of unique words of data 3
invert_3 = dict(map(reversed, word_index_3.items()))
vocab_size_3 = min(len(word_index_3)+1,(Num_Words))       #Number of unique tokens of data 3

#Preparing google word2vec embedding matrix for Data 3
embeddings_matrix_3 = np.zeros((vocab_size_3, Embedding_Dim))
for word, i in word_index_3.items():
	if i>= Num_Words:
		continue
	try:
		embedding_vector = word_vectors[word]
		embeddings_matrix_3[i] = embedding_vector
	except KeyError:
		embeddings_matrix_3[i] = np.random.normal(0, np.sqrt(0.25), Embedding_Dim)
		
		
#PREPARING Fasttext embedding for Data 3.
Embedding_Matrix_Fasttext_3 = np.zeros((vocab_size_3, Embedding_Dim))
for word, i in word_index_3.items():
	#print i,word
    	if i > Num_Words:
        	continue
	embedding_vector = embeddings_index_fasttext.get(word)
	if (embedding_vector is not None) and len(embedding_vector) > 0:
		Embedding_Matrix_Fasttext_3[i] = embedding_vector
	else:
		Embedding_Matrix_Fasttext_3[i] = model_ft.get_word_vector(word)

#concatenation of word2vec and fasttext embedding matrix for Data 3
concatenated_embedding_matrix_3 = np.hstack((embeddings_matrix_3,Embedding_Matrix_Fasttext_3))



#The count of unique tokens is assigned to "timer" variable
timer = Count
with open("D4.txt","r") as d:
	for line in d:
		line = line.lower()
		line = line.split()
		for item in line:
			if item not in Full_Dict:
				Full_Dict[item] = timer
				timer +=1



##Tokenize and padding the sequences for Data 4
with open("D4.txt","r")as d:
	texts_4 = d.readlines()
tokenizer_4 = Tokenizer(Num_Words)
tokenizer_4.fit_on_texts(texts_4)
Task_4 =  tokenizer_4.texts_to_sequences(texts_4)
data4 = pad_sequences(Task_4, maxlen=Max_length,padding = 'post')
word_index_4 = tokenizer_4.word_index               #dictionary of unique words of data 4
invert_4 = dict(map(reversed, word_index_4.items()))
vocab_size_4 = min(len(word_index_4)+1,(Num_Words))    #Number of unique tokens of data 4

#Preparing google word2vec embedding matrix for Data 4
embeddings_matrix_4 = np.zeros((vocab_size_4, Embedding_Dim))
for word, i in word_index_4.items():
	if i>= Num_Words:
		continue
	try:
		embedding_vector = word_vectors[word]
		embeddings_matrix_4[i] = embedding_vector
	except KeyError:
		embeddings_matrix_4[i] = np.random.normal(0, np.sqrt(0.25), Embedding_Dim)
		
		
#Preparing Fasttext embedding for Data 4.
Embedding_Matrix_Fasttext_4 = np.zeros((vocab_size_4, Embedding_Dim))
for word, i in word_index_4.items():
    	if i > Num_Words:
        	continue
	embedding_vector = embeddings_index_fasttext.get(word)
	if (embedding_vector is not None) and len(embedding_vector) > 0:
		Embedding_Matrix_Fasttext_4[i] = embedding_vector
	else:
		Embedding_Matrix_Fasttext_4[i] = model_ft.get_word_vector(word)
		

#concatenation of word2vec and fasttext embedding matrix for Data 4		
concatenated_embedding_matrix_4 = np.hstack((embeddings_matrix_4,Embedding_Matrix_Fasttext_4))




#The count of unique tokens from the 4 data is assigned to "number" variable
number = timer
with open("D5.txt","r") as e:
	for line in e:
		line = line.lower()
		line = line.split()
		for item in line:
			if item not in Full_Dict:
				Full_Dict[item] = number
				number +=1

Total = number + 1.       #This is the total number of unique tokens in the merged data
print "the value of Total unique tokens in all the data is",Total
print"The total element in Full_Dict is",len(Full_Dict)


##Tokenize and padding the sequences for Data 5
with open("D5.txt","r")as e:
	texts_5 = e.readlines()
tokenizer_5 = Tokenizer(Num_Words)
tokenizer_5.fit_on_texts(texts_5)
Task_5 =  tokenizer_5.texts_to_sequences(texts_5)
data5 = pad_sequences(Task_5, maxlen=Max_length,padding = 'post')
word_index_5 = tokenizer_5.word_index                #dictionary of unique words of data 5
invert_5 = dict(map(reversed, word_index_5.items()))
vocab_size_5 = min(len(word_index_5)+1,(Num_Words))      #Number of unique tokens of data 5


#Preparing google word2vec embedding matrix for Data 5
embeddings_matrix_5 = np.zeros((vocab_size_5, Embedding_Dim))
for word, i in word_index_5.items():
	if i>= Num_Words:
		continue
	try:
		embedding_vector = word_vectors[word]
		embeddings_matrix_5[i] = embedding_vector
	except KeyError:
		embeddings_matrix_5[i] = np.random.normal(0, np.sqrt(0.25), Embedding_Dim)
		

#Preparing Fasttext embedding for Data 5.
Embedding_Matrix_Fasttext_5 = np.zeros((vocab_size_5, Embedding_Dim))
for word, i in word_index_5.items():
    	if i > Num_Words:
        	continue
	embedding_vector = embeddings_index_fasttext.get(word)
	if (embedding_vector is not None) and len(embedding_vector) > 0:
		Embedding_Matrix_Fasttext_5[i] = embedding_vector
	else:
		Embedding_Matrix_Fasttext_5[i] = model_ft.get_word_vector(word)
		
#concatenation of word2vec and fasttext embedding matrix for Data 5	
concatenated_embedding_matrix_5 = np.hstack((embeddings_matrix_5,Embedding_Matrix_Fasttext_5))


"""
##Labels for the 2 Tasks
labels_task = pd.read_csv("Task_2_class.csv")
labels_task = labels_task['class'].values
labels_one_hot_task = np_utils.to_categorical(labels_task)
labels_1 = labels_one_hot_task[0:30]
labels_1_1 = labels_one_hot_task[0:29]
labels_2 = labels_one_hot_task[30:60]
labels_2_1 = labels_one_hot_task[30:59]
"""

"""
##Labels for the 3 Tasks
labels_task = pd.read_csv("Task_3_class.csv")
labels_task = labels_task['class'].values
labels_one_hot_task = np_utils.to_categorical(labels_task)
labels_1 = labels_one_hot_task[0:30]
labels_1_1 = labels_one_hot_task[0:29]
labels_2 = labels_one_hot_task[30:60]
labels_2_1 = labels_one_hot_task[30:59]
labels_3 = labels_one_hot_task[60:90]
labels_3_1 = labels_one_hot_task[60:89]
"""
"""
##Labels for the  4 Tasks
labels_task = pd.read_csv("Task_4_class.csv")
labels_task = labels_task['class'].values
labels_one_hot_task = np_utils.to_categorical(labels_task)
labels_task_1 = labels_one_hot_task[0:128]
labels_1 = labels_task_1[0:30]
labels_1_1 = labels_task_1[0:29]
labels_task_2 = labels_one_hot_task[128:256]
labels_2 = labels_task_2[0:30]
labels_2_1 = labels_task_2[0:29]
labels_task_3 = labels_one_hot_task[256:384]
labels_3 = labels_task_3[0:30]
labels_3_1 = labels_task_3[0:29]
labels_task_4 = labels_one_hot_task[384:414]
labels_4 = labels_task_4[0:30]
labels_4_1 = labels_task_4[0:29]
"""


###Labels for 5 classes
labels_task = pd.read_csv("Task_5_class.csv")
labels_task = labels_task['class'].values
labels_one_hot_task = np_utils.to_categorical(labels_task)
labels_task_1 = labels_one_hot_task[0:128]
labels_1 = labels_task_1[0:30]
labels_1_1 = labels_task_1[0:29]
labels_task_2 = labels_one_hot_task[128:256]
#print("The length of Task 2 is ",len(labels_task_2))
labels_2 = labels_task_2[0:30]
labels_2_1 = labels_task_2[0:29]
labels_task_3 = labels_one_hot_task[256:384]
labels_3 = labels_task_3[0:30]
labels_3_1 = labels_task_3[0:29]
labels_task_4 = labels_one_hot_task[384:414]
labels_4 = labels_task_4[0:30]
labels_4_1 = labels_task_4[0:29]
labels_task_5 = labels_one_hot_task[414:444]
labels_5 = labels_task_5[0:30]
labels_5_1 = labels_task_5[0:29]


#Preparing google word2vec shared Embedding Matrix for Whole Dataset
Embedding_Matrix = np.zeros((Total,Embedding_Dim))
for word,i in Full_Dict.items():
	if i>=Num_Words:
		continue
	try:
		embedding_vector = word_vectors[word]
		Embedding_Matrix[i] = embedding_vector
	except KeyError:
		Embedding_Matrix[i] = np.random.normal(0,np.sqrt(0.25),Embedding_Dim)

numberr = 1


#Preparing fasttext shared Embedding Matrix for Whole Dataset
Embedding_Matrix_Fasttext = np.zeros((Total, Embedding_Dim))
for word, i in Full_Dict.items():
    	if i > Num_Words:
        	continue
	embedding_vector = embeddings_index_fasttext.get(word)
	if (embedding_vector is not None) and len(embedding_vector) > 0:
		Embedding_Matrix_Fasttext[i] = embedding_vector
	else:
		Embedding_Matrix_Fasttext[i] = model_ft.get_word_vector(word)


#concatenation of word2vec and fasttext embedding matrix for Whole data
concatenated_embedding_matrix = np.hstack((Embedding_Matrix,Embedding_Matrix_Fasttext))



#This data will be used to chunk data as a batch size of 30 to train in shared space
count_1 = 1
Sequences_1 = []
with open("D1.txt","r") as d:
	for line in d:
		line = line.lower()
		line = line.split()
		for ind, item in enumerate(line):
			item = item.strip()
			if item in Full_Dict:
				line[ind] = Full_Dict[item]
		count_1 +=1
		Sequences_1.append(line)
data_1 = pad_sequences(Sequences_1, maxlen = Max_length, padding = 'post')



#This data will be used to chunk data as a batch size of 30 to train in shared space
count_2 = 1
Sequences_2 = []
with open("D2.txt") as e:
	for line in e:
		line = line.lower()
		line = line.split()
		for ind, item in enumerate(line):
			item = item.strip()
			if item in Full_Dict:
				line[ind] = Full_Dict[item]
		count_2 +=1
		Sequences_2.append(line)
data_2 = pad_sequences(Sequences_2, maxlen = Max_length, padding = 'post')




#This data will be used to chunk data as a batch size of 30 to train in shared space
count_3 = 1
Sequences_3 = []
with open("D3.txt","r")as f:
	for line in f:
		line = line.lower()
		line = line.split()
		for ind,item in enumerate(line):
			item = item.strip()
			if item in Full_Dict:
				line[ind] = Full_Dict[item]
		count_3 +=1
		Sequences_3.append(line)
data_3 = pad_sequences(Sequences_3, maxlen = Max_length, padding = 'post')



#This data will be used to chunk data as a batch size of 30 to train in shared space
count_4 = 1
Sequences_4 = []
with open("D4.txt","r") as g:
	for line in g:
		line = line.lower()
		line = line.split()
		for ind, item in enumerate(line):
			item = item.strip()
			if item in Full_Dict:
				line[ind] = Full_Dict[item]
		count_4 +=1
		Sequences_4.append(line)
data_4 = pad_sequences(Sequences_4, maxlen = Max_length, padding = 'post')



#This data will be used to chunk data as a batch size of 30 to train in shared space
count_5 = 1
Sequences_5 = []
with open("D5.txt","r") as h:
	for line in h:
		line = line.lower()
		line = line.split()
		for ind, item in enumerate(line):
			item = item.strip()
			if item in Full_Dict:
				line[ind] = Full_Dict[item]
		count_5 +=1
		Sequences_5.append(line)

data_5 = pad_sequences(Sequences_5, maxlen = Max_length, padding = 'post')



##Total number of sets with a size of 30
Steps_1 = math.ceil(len(data_1)/30.0)

##Total number of sets with a size of 30
Steps_2 = math.ceil(len(data_2)/30.0)

##Total number of sets with a size of 30
Steps_3 = math.ceil(len(data_3)/30.0)


##Total number of sets with a size of 30
Steps_4 = math.ceil(len(data_4)/30.0)

##Total number of sets with a size of 30
Steps_5 = math.ceil(len(data_5)/30.0)


###Dividing number of Training instances into Batches of 30
Data_1= np.array_split(data_1,Steps_1)

Data_2 =  np.array_split(data_2,Steps_2)

Data_3 =  np.array_split(data_3,Steps_3)

Data_4 =  np.array_split(data_4,Steps_4)

Data_5= np.array_split(data_5,Steps_5)


#1 Epoch is complete training of all the tweets. Here the epoch is 3 for the shared network
Steps = (int(max(Steps_1,Steps_2,Steps_3, Steps_4,Steps_5)))*3




#Class of Data 1 is converted to one hot encoding
Labels_1 = pd.read_csv("Data_1_class.csv")
Labels_1 = Labels_1['class'].values
Labels_one_hot_1 = np_utils.to_categorical(Labels_1)


#Class of Data 2 is converted to  one hot encoding
Labels_2 = pd.read_csv("Data_2_class.csv")
Labels_2 = Labels_2['class'].values
Labels_one_hot_2 = np_utils.to_categorical(Labels_2)

#Class of Data 3 is converted to  one hot encoding
Labels_3 = pd.read_csv("Data_3_class.csv")
Labels_3 = Labels_3['class'].values
Labels_one_hot_3 = np_utils.to_categorical(Labels_3)


#Class of Data 4 is converted to  one hot encoding
Labels_4 = pd.read_csv("Data_4_class.csv")
Labels_4 = Labels_4['class'].values
Labels_one_hot_4 = np_utils.to_categorical(Labels_4)


#Class of Data 5 is converted to one hot encoding
Labels_5 = pd.read_csv("Data_5_class.csv")
Labels_5 = Labels_5['class'].values
Labels_one_hot_5 = np_utils.to_categorical(Labels_5)



#Defining the shared neural network for 5 tasks trained jointly
input_1 = Input(shape = (Max_length,),dtype = 'int32')
embedding = Embedding(Total, 600, weights = [concatenated_embedding_matrix],input_length = 40, mask_zero = False,trainable = True)(input_1)
unigram = Conv1D(filters = 100, kernel_size = 1, padding = 'same', activation = 'relu',strides = 1)(embedding)
unigram = MaxPooling1D(pool_size = 4)(unigram)
bigram = Conv1D(filters = 100, kernel_size = 2, padding = 'same', activation = 'relu', strides = 1)(embedding)
bigram = MaxPooling1D(pool_size = 4)(bigram)
trigram = Conv1D(filters = 100, kernel_size = 3, padding = 'same', activation = 'relu', strides = 1)(embedding)
trigram = MaxPooling1D(pool_size = 4)(trigram)
fourgram = Conv1D(filters = 100, kernel_size = 4, padding = 'same', activation = 'relu', strides = 1)(embedding)
fourgram = MaxPooling1D(pool_size = 4)(fourgram)
merged = concatenate([unigram,bigram,trigram,fourgram])
merged = Flatten()(merged)
Prediction_1 = Dense(5,activation = 'softmax')(merged)
Model_1 = Model(inputs = [input_1],outputs = [Prediction_1])
#print (Model_1.summary())
Model_1.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])




# 5 classification tasks are trained jointly to capture task specific features. The batch size of 30 is utilized and last chunk may contain a size of 29 tweets. If 2 tasks are trained then\
# uncomment the line from 348-357. For 3 tasks uncomment the line from 359-370. For 4 tasks uncomment line 372-388.
for i in range(0, Steps):
	X = Data_1[i%len(Data_1)]    #Data 1
	Y = Data_2[i%len(Data_2)]    #Data 2
	Z = Data_3[i%len(Data_3)]    #Data 3
	A = Data_4[i%len(Data_4)]     #Data 4
	P = Data_5[i%len(Data_5)]     #Data 5
	if len(X) == 30 :
		x = labels_1
	else:
		x = labels_1_1
	if len(Y) == 30:
		y = labels_2
	else:
		y = labels_2_1
	if len(Z) == 30:
		z = labels_3
	else:
		z = labels_3_1

	if len(A) == 30:
		a = labels_4
	else:
		a = labels_4_1
	if len(P) == 30 :
		p = labels_5
	else:
		p = labels_5_1
		
		
		
	Model_1.train_on_batch(Data_1[i%len(Data_1)],x,sample_weight = None, class_weight = None)
	Model_1.train_on_batch(Data_2[i%len(Data_2)],y,sample_weight = None, class_weight = None)
	Model_1.train_on_batch(Data_3[i%len(Data_3)],z,sample_weight = None, class_weight = None)
	Model_1.train_on_batch(Data_4[i%len(Data_4)],a,sample_weight = None, class_weight = None)
	Model_1.train_on_batch(Data_5[i%len(Data_5)],p,sample_weight = None, class_weight = None)
	print (i)
	print (Model_1.summary())

Model_1.save("T1T2T3T4T5_CNN.hdf5")     #Saving the model weights

Model_1.load_weights("T1T2T3T4T5_CNN.hdf5")   #Load the trained model weight
shared_model = Model(inputs = Model_1.input,outputs = Model_1.layers[11].output)   #Sliced layers from trained Neural network to transfer shared knowledge
print (shared_model.summary())


file_1 = 1
file_2 = 6
file_3 = 11
file_4 = 16
file_5 = 21
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores_1 = []
z_1 = np.zeros([3,3],dtype= int)      #Data with 3 class
score_array_1 = []
cvscores_2 = []
z_2 = np.zeros([3,3],dtype= int)     #Data with 3 class
score_array_2 = [] 
cvscores_3 = []
z_3 = np.zeros([3,3],dtype= int)     #Data with 3 class
score_array_3 = []
cvscores_4 = []
z_4 = np.zeros([2,2],dtype= int)     #Data with 2 class
score_array_4 = []
cvscores_5 = []
z_5 = np.zeros([2,2],dtype= int)      #Data with 2 class
score_array_5 = []



#5 Fold cross validation for Data 1
for train,test in kfold.split(data1, Labels_1):
	input_2 = Input(shape = (Max_length,),dtype = 'int32')
	embedding = Embedding(vocab_size_1, 600, weights = [concatenated_embedding_matrix_1],input_length = 40, mask_zero = False,trainable = True)(input_2)
	unigram = Conv1D(filters = 100, kernel_size = 1, padding = 'same', activation = 'relu',strides = 1)(embedding)
	unigram = MaxPooling1D(pool_size = 4)(unigram)
	bigram = Conv1D(filters = 100, kernel_size = 2, padding = 'same', activation = 'relu', strides = 1)(embedding)
	bigram = MaxPooling1D(pool_size = 4)(bigram)
	trigram = Conv1D(filters = 100, kernel_size = 3, padding = 'same', activation = 'relu', strides = 1)(embedding)
	trigram = MaxPooling1D(pool_size = 4)(trigram)
	fourgram = Conv1D(filters = 100, kernel_size = 4, padding = 'same', activation = 'relu', strides = 1)(embedding)
	fourgram = MaxPooling1D(pool_size = 4)(fourgram)
	merged = concatenate([unigram,bigram,trigram,fourgram])
	merged = Flatten()(merged)
	shared_features = shared_model(input_2)
	merged_1 = concatenate([merged,shared_features])
	Predict_Task = Dense(3,activation = 'softmax')(merged_1)
	model_task_1 = Model(inputs = input_2, outputs = Predict_Task)
	model_task_1.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
	print (model_task_1.summary())
	print (model_task_1.layers)
	filepath="T1.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	model_task_1.fit(data1[train], Labels_one_hot_1[train],validation_data = (data1[test],Labels_one_hot_1[test]),callbacks=callbacks_list,epochs=3, batch_size=16, verbose=2)
	#model_task_1.fit(data1[train], Labels_one_hot_1[train],validation_split = 0.1,callbacks=callbacks_list,epochs=3, batch_size=16, verbose=2)
	model_task_1.save("T1.hdf5")
	model_task_1 = load_model("T1.hdf5")
	scores = model_task_1.evaluate(data1[test], Labels_one_hot_1[test],batch_size=16)
	cvscores_1.append(scores[1] * 100)
	new_prediction = model_task_1.predict(data1[test])
	print new_prediction
	new_prediction_1 = new_prediction.argmax(axis= -1)
	original_prediction = np.argmax(Labels_one_hot_1[test], axis=1)
	print original_prediction
	sentence = []
	# #CSV file containing senetnce, actual prediction and predicted prediction for the analysis
	for line in data1[test]:
		list = []
		for a in line:
			if a in invert_1:
				a = invert_1[a]
				list.append(a)
		a = " ".join(list)
		sentence.append(a)
	test_data = np.asarray(data1[test])
	#print test_data
	df = pd.DataFrame(columns=['tweets', 'real_label', 'new_label'])
	df['tweets'] = sentence
	df['real_label'] = original_prediction
	df['new_label'] = new_prediction_1
	df.to_csv('new_prediction' + str(file_1) + ".csv", sep=',')      #CSV file containing sentences, actual tags and predicted tags
	cm1 = confusion_matrix(np.argmax(Labels_one_hot_1[test],axis = 1),np.argmax(new_prediction,axis = 1))
	precision_recall = precision_recall_fscore_support(original_prediction,np.argmax(new_prediction,axis = 1))
	score_array_1.append(precision_recall_fscore_support(original_prediction,np.argmax(new_prediction,axis = 1), average=None))
	print cm1    # print confusion matrix for each fold
	z_1 = cm1+z_1   #adding all the 5 fold confusion matrix
print z_1   #Final confusion matrix for Data 1
a = float(z_1[0][0])
b = float(z_1[1][0])
c = float(z_1[2][0])
d = float(z_1[0][1])
e = float(z_1[1][1])
f = float(z_1[2][1])
g = float(z_1[0][2])
h = float(z_1[1][2])
i = float(z_1[2][2])
P1 = ((a)/(a+b+c))      #Precision class 1
R1 = ((a)/(a+d+g))      #Recall class 1
F1 = ((2*P1*R1)/(P1+R1))  #F-score class 1
P2 = ((e)/(d+e+f))       #Precision class 2
R2 = ((e)/(b+e+h))       #Recall class 2
F2 = ((2*P2*R2)/(P2+R2))  #F-score class 2
P3 = ((i)/(g+h+i))      #Precision class 3
R3 = ((i)/(c+f+i))       #Recall class 3
F3 = ((2*P3*R3)/(P3+R3))   #F-score class 3
Macro_P= (P1+P2+P3)/3    #Macro Precision 
Macro_R = (R1+R2+R3)/3   #Macro Recall 
Macro_F1 = (F1+F2+F3)/3   #Macro-F score
print ("The precision for class a is",P1)
print ("The recall for class a is",R1)
print ("The fscore for class a is",F1)
print ("The precision for class b is",P2)
print ("The recall for class b is",R2)
print ("The fscore for class b is",F2)
print ("The precision for class c is",P3)
print ("The recall for class c is",R3)
print ("The fscore for class c is",F3)
print("Macro_P",Macro_P)
print("Macro_R",Macro_R)
print("Macro_F",Macro_F1)





#5 Fold cross validation for Data 2
for train,test in kfold.split(data2, Labels_2):
	input_2 = Input(shape = (Max_length,),dtype = 'int32')
	embedding = Embedding(vocab_size_2, 600, weights = [concatenated_embedding_matrix_2],input_length = 40, mask_zero = False,trainable = True)(input_2)
	unigram = Conv1D(filters = 100, kernel_size = 1, padding = 'same', activation = 'relu',strides = 1)(embedding)
	unigram = MaxPooling1D(pool_size = 4)(unigram)
	bigram = Conv1D(filters = 100, kernel_size = 2, padding = 'same', activation = 'relu', strides = 1)(embedding)
	bigram = MaxPooling1D(pool_size = 4)(bigram)
	trigram = Conv1D(filters = 100, kernel_size = 3, padding = 'same', activation = 'relu', strides = 1)(embedding)
	trigram = MaxPooling1D(pool_size = 4)(trigram)
	fourgram = Conv1D(filters = 100, kernel_size = 4, padding = 'same', activation = 'relu', strides = 1)(embedding)
	fourgram = MaxPooling1D(pool_size = 4)(fourgram)
	merged = concatenate([unigram,bigram,trigram,fourgram])
	merged = Flatten()(merged)
	shared_features = shared_model(input_2)
	merged_1 = concatenate([merged,shared_features])
	Predict_Task = Dense(3,activation = 'softmax')(merged_1)
	model_task_2 = Model(inputs = input_2, outputs = Predict_Task)
	model_task_2.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
	print (model_task_2.summary())
	print (model_task_2.layers)
	filepath="T2.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	model_task_2.fit(data2[train], Labels_one_hot_2[train],validation_data = (data2[test],Labels_one_hot_2[test]),callbacks=callbacks_list,epochs=3, batch_size=16, verbose=2)
	#model_task_2.fit(data2[train], Labels_one_hot_2[train],validation_split = 0.1,callbacks=callbacks_list,epochs=3, batch_size=16, verbose=2)
	model_task_2.save("T2.hdf5")
	model_task_2 = load_model("T2.hdf5")
	scores = model_task_2.evaluate(data2[test], Labels_one_hot_2[test],batch_size=16)
	cvscores_2.append(scores[1] * 100)
	new_prediction = model_task_2.predict(data2[test])
	print (new_prediction)
	new_prediction_1 = new_prediction.argmax(axis= -1)
	original_prediction = np.argmax(Labels_one_hot_2[test], axis=1)
	print (original_prediction)
	sentence = []
	#CSV file containing senetnce, actual prediction and predicted prediction for the analysis
	for line in data2[test]:
		#print line
		list = []
		for a in line:
			if a in invert_2:
				a = invert_2[a]
				list.append(a)
		a = " ".join(list)
		sentence.append(a)
	test_data = np.asarray(data2[test])
	#print test_data
	df = pd.DataFrame(columns=['tweets', 'real_label', 'new_label'])
	df['tweets'] = sentence
	df['real_label'] = original_prediction
	df['new_label'] = new_prediction_1
	df.to_csv('new_prediction' + str(file_2) + ".csv", sep=',')      #CSV file containing sentences, actual tags and predicted tags
	cm2 = confusion_matrix(np.argmax(Labels_one_hot_2[test],axis = 1),np.argmax(new_prediction,axis = 1))
	precision_recall = precision_recall_fscore_support(original_prediction,np.argmax(new_prediction,axis = 1))
	score_array_2.append(precision_recall_fscore_support(original_prediction,np.argmax(new_prediction,axis = 1), average=None))
	print cm2      # print confusion matrix for each fold
	z_2 = cm2+z_2     #adding all the 5 fold confusion matrix
print z_2         #Final confusion matrix for Data 1
a = float(z_2[0][0])
b = float(z_2[1][0])
c = float(z_2[2][0])
d = float(z_2[0][1])
e = float(z_2[1][1])
f = float(z_2[2][1])
g = float(z_2[0][2])
h = float(z_2[1][2])
i = float(z_2[2][2])
P1 = (a/(a+b+c))      #Precision class 1
R1 = (a/(a+d+g))      #Recall class 1
F1 = ((2*P1*R1)/(P1+R1))  #F-score class 1
P2 = (e/(d+e+f))       #Precision class 2
R2 = (e/(b+e+h))       #Recall class 2
F2 = ((2*P2*R2)/(P2+R2))  #F-score class 2
P3 = (i/(g+h+i))      #Precision class 3
R3 = (i/(c+f+i))       #Recall class 3
F3 = ((2*P3*R3)/(P3+R3))   #F-score class 3
Macro_P= (P1+P2+P3)/3    #Macro Precision 
Macro_R = (R1+R2+R3)/3   #Macro Recall 
Macro_F1 = (F1+F2+F3)/3   #Macro-F score
print ("The precision for class a is",P1)
print ("The recall for class a is",R1)
print ("The fscore for class a is",F1)
print ("The precision for class b is",P2)
print ("The recall for class b is",R2)
print ("The fscore for class b is",F2)
print ("The precision for class c is",P3)
print ("The recall for class c is",R3)
print ("The fscore for class c is",F3)
print("Macro_P",Macro_P)
print("Macro_R",Macro_R)
print("Macro_F",Macro_F1)




#5 Fold cross validation for Data 3
for train,test in kfold.split(data3, Labels_3):
	input_2 = Input(shape = (Max_length,),dtype = 'int32')
	embedding = Embedding(vocab_size_3, 600, weights = [concatenated_embedding_matrix_3],input_length = 40, mask_zero = False, trainable = True)(input_2)
	unigram = Conv1D(filters = 100, kernel_size = 1, padding = 'same', activation = 'relu',strides = 1)(embedding)
	unigram = MaxPooling1D(pool_size = 4)(unigram)
	bigram = Conv1D(filters = 100, kernel_size = 2, padding = 'same', activation = 'relu', strides = 1)(embedding)
	bigram = MaxPooling1D(pool_size = 4)(bigram)
	trigram = Conv1D(filters = 100, kernel_size = 3, padding = 'same', activation = 'relu', strides = 1)(embedding)
	trigram = MaxPooling1D(pool_size = 4)(trigram)
	fourgram = Conv1D(filters = 100, kernel_size = 4, padding = 'same', activation = 'relu', strides = 1)(embedding)
	fourgram = MaxPooling1D(pool_size = 4)(fourgram)
	merged = concatenate([unigram,bigram,trigram,fourgram])
	merged = Flatten()(merged)
	shared_features = shared_model(input_2)
	merged_1 = concatenate([merged,shared_features])
	#merged_1 = Flatten(merged_1)
	Predict_Task = Dense(3, activation = 'softmax')(merged_1)
	model_task_3= Model(inputs = input_2, outputs = Predict_Task)
	model_task_3.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
	print (model_task_3.summary())
	print (model_task_3.layers)
	filepath="T3.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	model_task_3.fit(data3[train], Labels_one_hot_3[train],validation_data = (data3[test],Labels_one_hot_3[test]),callbacks=callbacks_list,epochs=3, 		batch_size=16, verbose=2)
	#model_task_3.fit(data3[train], Labels_one_hot_3[train],validation_split = 0.1,callbacks=callbacks_list,epochs=3, batch_size=16, verbose=2)
	model_task_3.save("T3.hdf5")
	model_task_3 = load_model("T3.hdf5")
	scores = model_task_3.evaluate(data3[test], Labels_one_hot_3[test],batch_size=16)
	cvscores_3.append(scores[1] * 100)
	new_prediction = model_task_3.predict(data3[test])
	print (new_prediction)
	new_prediction_1 = new_prediction.argmax(axis= -1)
	original_prediction = np.argmax(Labels_one_hot_3[test], axis=1)
	print (original_prediction)
	sentence = []
	#CSV file containing senetnce, actual prediction and predicted prediction for the analysis
	for line in data3[test]:
		list = []
		for a in line:
			if a in invert_3:
				a = invert_3[a]
				list.append(a)
		a = " ".join(list)
		sentence.append(a)
	test_data = np.asarray(data3[test])
	#print test_data
	df = pd.DataFrame(columns=['tweets', 'real_label', 'new_label'])
	df['tweets'] = sentence
	df['real_label'] = original_prediction
	df['new_label'] = new_prediction_1
	df.to_csv('new_prediction' + str(file_3) + ".csv", sep=',')        #CSV file containing sentences, actual tags and predicted tags
	file_3+=1
	cm3 = confusion_matrix(np.argmax(Labels_one_hot_3[test],axis = 1),np.argmax(new_prediction,axis = 1))
	print cm3                # print confusion matrix for each fold
	z_3 = cm3+z_3             #adding all the 5 fold confusion matrix
print z_3                   #Final confusion matrix for Data 3
a = float(z_3[0][0])
b = float(z_3[1][0])
c = float(z_3[2][0])
d = float(z_3[0][1])
e = float(z_3[1][1])
f = float(z_3[2][1])
g = float(z_3[0][2])
h = float(z_3[1][2])
i = float(z_3[2][2])
P1 = (a/(a+b+c))      #Precision class 1
R1 = (a/(a+d+g))      #Recall class 1
F1 = ((2*P1*R1)/(P1+R1))  #F-score class 1
P2 = (e/(d+e+f))       #Precision class 2
R2 = (e/(b+e+h))       #Recall class 2
F2 = ((2*P2*R2)/(P2+R2))  #F-score class 2
P3 = (i/(g+h+i))      #Precision class 3
R3 = (i/(c+f+i))       #Recall class 3
F3 = ((2*P3*R3)/(P3+R3))   #F-score class 3
Macro_P= (P1+P2+P3)/3    #Macro Precision 
Macro_R = (R1+R2+R3)/3   #Macro Recall 
Macro_F1 = (F1+F2+F3)/3   #Macro-F score
print ("The precision for class a is",P1)
print ("The recall for class a is",R1)
print ("The fscore for class a is",F1)
print ("The precision for class b is",P2)
print ("The recall for class b is",R2)
print ("The fscore for class b is",F2)
print ("The precision for class c is",P3)
print ("The recall for class c is",R3)
print ("The fscore for class c is",F3)
print("Macro_P",Macro_P)
print("Macro_R",Macro_R)
print("Macro_F",Macro_F)





#5 Fold cross validation for Data 4
for train,test in kfold.split(data4, Labels_4):
	input_2 = Input(shape = (Max_length,),dtype = 'int32')
	embedding = Embedding(vocab_size_4, 600, weights = [concatenated_embedding_matrix_4],input_length = 40, mask_zero = False,trainable = True)(input_2)
	unigram = Conv1D(filters = 100, kernel_size = 1, padding = 'same', activation = 'relu',strides = 1)(embedding)
	unigram = MaxPooling1D(pool_size = 4)(unigram)
	bigram = Conv1D(filters = 100, kernel_size = 2, padding = 'same', activation = 'relu', strides = 1)(embedding)
	bigram = MaxPooling1D(pool_size = 4)(bigram)
	trigram = Conv1D(filters = 100, kernel_size = 3, padding = 'same', activation = 'relu', strides = 1)(embedding)
	trigram = MaxPooling1D(pool_size = 4)(trigram)
	fourgram = Conv1D(filters = 100, kernel_size = 4, padding = 'same', activation = 'relu', strides = 1)(embedding)
	fourgram = MaxPooling1D(pool_size = 4)(fourgram)
	merged = concatenate([unigram,bigram,trigram,fourgram])
	merged = Flatten()(merged)
	shared_features = shared_model(input_2)
	merged_1 = concatenate([merged,shared_features])
	Predict_Task = Dense(2,activation = 'softmax')(merged_1)
	model_task_4= Model(inputs = input_2, outputs = Predict_Task)
	model_task_4.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
	print (model_task_4.summary())
	print (model_task_4.layers)
	filepath="T4.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	model_task_4.fit(data4[train], Labels_one_hot_4[train],validation_data = (data4[test],Labels_one_hot_4[test]),callbacks=callbacks_list,epochs=3, 		batch_size=16, verbose=2)
	#model_task_4.fit(data4[train], Labels_one_hot_4[train],validation_split = 0.1,callbacks=callbacks_list,epochs=3, batch_size=16, verbose=2)
	model_task_4.save("T4.hdf5")
	model_task_4 = load_model("T4.hdf5")
	scores = model_task_4.evaluate(data4[test], Labels_one_hot_4[test],batch_size=16)
	cvscores_4.append(scores[1] * 100)
	new_prediction = model_task_4.predict(data4[test])
	print (new_prediction)
	new_prediction_1 = new_prediction.argmax(axis= -1)
	original_prediction = np.argmax(Labels_one_hot_4[test], axis=1)
	print (original_prediction)
	sentence = []
	#CSV file containing senetnce, actual prediction and predicted prediction for the analysis
	for line in data4[test]:
		#print line
		list = []
		for a in line:
			if a in invert_4:
				a = invert_4[a]
				list.append(a)
		a = " ".join(list)
		sentence.append(a)
	test_data = np.asarray(data4[test])
	#print test_data
	df = pd.DataFrame(columns=['tweets', 'real_label', 'new_label'])
	df['tweets'] = sentence
	df['real_label'] = original_prediction
	df['new_label'] = new_prediction_1
	df.to_csv('new_prediction' + str(file_4) + ".csv", sep=',')         #CSV file containing sentences, actual tags and predicted tags
	file_4+=1
	cm4 = confusion_matrix(np.argmax(Labels_one_hot_4[test],axis = 1),np.argmax(new_prediction,axis = 1))
	precision_recall = precision_recall_fscore_support(original_prediction,np.argmax(new_prediction,axis = 1))
	score_array_4.append(precision_recall_fscore_support(original_prediction,np.argmax(new_prediction,axis = 1), average=None))
	print cm4            # print confusion matrix for each fold
	z_4 = cm4+z_4          #adding all the 5 fold confusion matrix
print z_4                #Final confusion matrix for Data 4
p = float(z_4[0][0])
q = float(z_4[0][1])
r = float(z_4[1][0])
s = float(z_4[1][1])
PR1 = (p/(p+r))
RE1 = (p/(p+q))
FS1 = (2*PR1*RE1)/(PR1+RE1)
PR2 = (s/(q+s))
RE2 = (s/(r+s))
FS2 = (2*PR2*RE2)/(PR2+RE2)
Macro_P = (PR1+PR2)/2
Macro_R = (RE1+RE2)/2
Macro_F1 = (FS1+FS2)/2
print ("The precision for class a is",PR1)
print ("The recall for class a is",RE1)
print ("The fscore for class a is",FS1)
print ("The precision for class b is",PR2)
print ("The recall for class b is",RE2)
print ("The f-scorefor class b is",FS2)
print("Macro_F",Macro_F1)




#5 Fold cross validation for Data 5
for train,test in kfold.split(data5, Labels_5):
	input_2 = Input(shape = (Max_length,),dtype = 'int32')
	embedding = Embedding(vocab_size_5, 600, weights = [concatenated_embedding_matrix_5],input_length = 40, mask_zero = False,trainable = True)(input_2)
	unigram = Conv1D(filters = 100, kernel_size = 1, padding = 'same', activation = 'relu',strides = 1)(embedding)
	unigram = MaxPooling1D(pool_size = 4)(unigram)
	bigram = Conv1D(filters = 100, kernel_size = 2, padding = 'same', activation = 'relu', strides = 1)(embedding)
	bigram = MaxPooling1D(pool_size = 4)(bigram)
	trigram = Conv1D(filters = 100, kernel_size = 3, padding = 'same', activation = 'relu', strides = 1)(embedding)
	trigram = MaxPooling1D(pool_size = 4)(trigram)
	fourgram = Conv1D(filters = 100, kernel_size = 4, padding = 'same', activation = 'relu', strides = 1)(embedding)
	fourgram = MaxPooling1D(pool_size = 4)(fourgram)
	merged = concatenate([unigram,bigram,trigram,fourgram])
	merged = Flatten()(merged)
	shared_features = shared_model(input_2)
	merged_1 = concatenate([merged, shared_features])
	Predict_Task = Dense(2,activation = 'softmax')(merged_1)
	model_task_5= Model(inputs = input_2, outputs = Predict_Task)
	model_task_5.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
	print (model_task_5.summary())
	print (model_task_5.layers)
	filepath="T5.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	model_task_5.fit(data5[train], Labels_one_hot_5[train],validation_data = (data5[test],Labels_one_hot_5[test]),callbacks=callbacks_list,epochs=3, 		batch_size=16, verbose=2)
	#model_task_5.fit(data5[train], Labels_one_hot_5[train],validation_split = 0.1,callbacks=callbacks_list,epochs=3, batch_size=16, verbose=2)
	model_task_5.save("T5.hdf5")
	model_task_5 = load_model("T5.hdf5")
	scores = model_task_5.evaluate(data5[test], Labels_one_hot_5[test],batch_size=16)
	cvscores_5.append(scores[1] * 100)
	new_prediction = model_task_5.predict(data5[test])
	print (new_prediction)
	new_prediction_1 = new_prediction.argmax(axis= -1)
	original_prediction = np.argmax(Labels_one_hot_5[test], axis=1)
	print (original_prediction)
	sentence = []
	#CSV file containing senetnce, actual prediction and predicted prediction for the analysis
	for line in data5[test]:
		#print line
		list = []
		for a in line:
			if a in invert_5:
				a = invert_5[a]
				list.append(a)
		a = " ".join(list)
		sentence.append(a)
	test_data = np.asarray(data5[test])
	#print test_data
	df = pd.DataFrame(columns=['tweets', 'real_label', 'new_label'])
	df['tweets'] = sentence
	df['real_label'] = original_prediction
	df['new_label'] = new_prediction_1
	df.to_csv('new_prediction' + str(file_5) + ".csv", sep=',')      #CSV file containing sentences, actual tags and predicted tags
	file_5+=1
	cm5 = confusion_matrix(np.argmax(Labels_one_hot_5[test],axis = 1),np.argmax(new_prediction,axis = 1))
	precision_recall = precision_recall_fscore_support(original_prediction,np.argmax(new_prediction,axis = 1))
	score_array_5.append(precision_recall_fscore_support(original_prediction,np.argmax(new_prediction,axis = 1), average=None))
	print cm5         # print confusion matrix for each fold
	z_5 = cm5+z_5       #adding all the 5 fold confusion matrix
print z_5             #Final confusion matrix for Data 5
p = float(z_5[0][0])
q = float(z_5[0][1])
r = float(z_5[1][0])
s = float(z_5[1][1])
PR1 = (p/(p+r))
RE1 = (p/(p+q))
FS1 = (2*PR1*RE1)/(PR1+RE1)
PR2 = (s/(q+s))
RE2 = (s/(r+s))
FS2 = (2*PR2*RE2)/(PR2+RE2)
Macro_P = (PR1+PR2)/2
Macro_R = (RE1+RE2)/2
Macro_F1 = (FS1+FS2)/2
print ("The precision for class a is",PR1)
print ("The recall for class a is",RE1)
print ("The fscore for class a is",FS1)
print ("The precision for class b is",PR2)
print ("The recall for class b is",RE2)
print ("The f-score for class b is",FS2)
print("Macro_F",Macro_F1)








