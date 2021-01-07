import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
import tkinter
import read_func
import re

height=30
width=60
num=100

batch_size=30
num_words=tf.Variable(1,dtype='int32',trainable=False)
vocab=read_func.build_vocab('ptb.train.txt')
model=tf.keras.models.load_model('rnn_model',compile=False)


def new(ids,b=batch_size,n=num_words):
    arr=np.zeros((b,n.numpy()))
    arr[0,:]=ids
    return arr

def read(txt):
    global vocab
    words=txt.strip().lower().split()
    ids=read_func.word_to_id(words,vocab)
    num_words.assign(len(ids))
    input_arr=new(ids)
    return input_arr

def predict(arr):
    y_hat=model(arr)
    ids=np.argmax(y_hat[0],axis=1)
    words=read_func.id_to_word(ids,vocab)
    return words

def extract(event):
    global num
    txt=text.get(index1=f'insert-{num}c wordstart',index2='insert')
    eos=re.findall("([a-z]+\s*[.?])",txt)
    
    if len(eos)!=0:
        eos_ind=re.search(eos[-1],txt).span()[1]
    else:
        eos_ind=0
        
    txt=txt[eos_ind:]
    input_arr=read(txt)
    
    try: 
        predicted=predict(input_arr)[-1]
    except:
        predicted=''
    
    if predicted=='<eos>':
        output='.'
    elif predicted=='<unk>':
        output=''
    else:
        output=predicted
    
    label['text']=output
    
def insert(event):
    text.insert('end',label['text'])

window=tkinter.Tk()
window.title('Composer')
frame = tkinter.Frame(window)
frame.pack() 
text=tkinter.Text(frame,height=height,width=width)
text.pack()
text.bind("<space>",extract,False)
text.bind("<Right>",insert,False)
label = tkinter.Label(frame,text='Start writing...',width=num,height=3)
label.pack()
window.mainloop()

