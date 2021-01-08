curl "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz" -o "data.tgz"
tar -cvzf "data.tgz"
pip install numpy==1.19.4
pip install tensorflow==2.3.1
pip install grpcio==1.32.0
pip install tk