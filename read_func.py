import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import collections

path='./simple-examples/data'

def read(filename,path):
    with tf.io.gfile.GFile(os.path.join(path,filename)) as file:
        data=file.read().replace('\n','<eos>').split()
    return data

def build_vocab(filename):
    data=read(filename,path)
    counter=collections.Counter()
    counter.update(data)
    sort=sorted(counter.items(),key=lambda x:(-x[1],x[0]))
    words,counts= zip(*sort)
    vocab=dict(zip(words,range(len(words))))
    return vocab

def word_to_id(data,vocab):
    return [vocab[word] for word in data if word in vocab]

def id_to_word(ids,vocab):
    id_to_word=[]
    for id_ in ids:
        for word in vocab.keys():
            if id_==vocab[word]:
                id_to_word.append(word)
                break
    return id_to_word

def data_to_batch(data,batch_size,time_steps):
    batch_len=len(data)//batch_size
    batch=np.zeros((batch_size,batch_len),dtype=int)
    for i in range(0,batch_size):
        batch[i]=data[i*batch_len:(i+1)*batch_len]
    
    epoch=(batch_len-1)//20
    for i in range(0,epoch):
        x=batch[:,i*time_steps:(i+1)*time_steps]
        y=batch[:,i*time_steps+1:(i+1)*time_steps+1]
        yield x,y
        
def reader(filename,vocab):
    read_dat=read(filename,path)
    data=word_to_id(read_dat,vocab)
    return data
