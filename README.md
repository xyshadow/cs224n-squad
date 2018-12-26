# cs224n-squad
cs224n squad project

Starter code from Stanford CS224n: http://web.stanford.edu/class/cs224n/default_project/index.html

The work being done in this repo:
1. Update starter code to be compatible with tensorflow 1.12
2. Convert GRU to cudnn_rnn.CudnnGRU & cudnn_rnn.CudnnLSTM, no recurent drop out support, but massive improvement on run time.
3. Implemented Bidirectional attention & Coattention
4. Created an attention called RNNattention, with RNNattention, simply feed the attention output to another bidirectional RNN, the model is able to achieve 65%+ F1 score on dev set with extremly fast training time (leave every other parameters as default). Training time per batch on a single NV 1080 graphic card machine is 0.26s.

Things to do:
1. Look at charactor-level vector representation
2. Look at conditioning end prediction on start prediction
