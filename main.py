from flask import Flask,request,render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
import reader
import json

app=Flask(__name__)

#--------------loading data--------------------

with open('./vocab.json','r') as file:
    vocab=file.read()
file.close()
vocab=json.loads(vocab)

#--------------loading model--------------------

with open('./model_config.json','r') as file:
    config=file.read()
file.close()
model=keras.models.model_from_json(config)
model.load_weights('./weights.h5')


#------------defining functions-----------------

context=[]
num=1
def predict(words,time_steps,num):
    for i in range(0,num):
        inp=words[-time_steps:]
        inp=reader.word_to_id(inp,vocab)
        inp=np.reshape(inp,(1,time_steps,1))
        out=model.predict(inp)
        out=np.argmax(out,axis=2)[0]
        results=reader.id_to_word(out,vocab)
        words.append(results[-1])
    return words

#-------------defining events-----------------

@app.route('/',methods=['GET'])
def welcome():
    return render_template('index.html')


@app.route('/context',methods=['post'])
def get_context(lst=context):
    if request.form.get('submit'):
        data=request.form.get('data')
        data=data.replace('.','<eos>')
        lst.extend(data.split())
        string='Context submitted!'
    else:
        lst[:]=[]
        string='Context empty!'
    return render_template('index.html',status=string)


@app.route('/words',methods=['post'])
def get_num():
    global num
    data=request.form.get('length')
    if data:
        num=int(data)
        return render_template('index.html',status=f'Ready to generate {num} words')
    else:
        return render_template('index.html',status="Invalid entry")


@app.route('/predict',methods=['post'])
def one(lst=context):
    global num
    if request.form.get('one'):
        results=predict(lst,20,1)
    elif request.form.get('custom'):
        results=predict(lst,20,num)
    else:
        results=['More','context','needed!']
    results=' '.join(results).replace('<eos>','.')
    return render_template('index.html',output=results,status='Conpleted!')

#-------------------main------------------------

if __name__ == '__main__':
    app.run(port=8000,host='127.0.0.1',debug=True)
