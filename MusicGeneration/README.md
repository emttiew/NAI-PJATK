## Academic project for music generation 

Generate single instrument music in midi format using neural networks using 3 models LSTM, SimpleRNN or WaveNet

Authors Mateusz Woźniak 18182, Jakub Włoch 16912

### Requirements
  - Python 3.7
  - Tensorflow
  - Music21
  - Keras
  - H5py
  - Numpy
  - Some midi player

## Learning curve
### LSTM
![](https://github.com/emttiew/NAI-PJATK/blob/master/MusicGeneration/data/plots/lstm.png?raw=true)
### SimpleRNN
![](https://github.com/emttiew/NAI-PJATK/blob/master/MusicGeneration/data/plots/simple_rnn.png?raw=true)
## Usage

### Training

Dataset with midi songs is in data/schubert folder. You can upload your own midi files and train chosen model by typing:
- `python train_lstm.py`
- `python train_simple_rnn.py`
- `python train_wavenet.py` (not quite working model)

Weights with parameters to newly generated model will be saved in data/weights directory

### Generate

To generate music using chosen pre-trained model type:
- `python generate_lstm.py`
- `python generate_simple_rnn.py`
- `python generate_wavenet.py` 

If you want to use your newly trained model change this line in code (in `create_network` function) with the name of generated weight from the previous point:
`model.load_weights('data/weights/your-new-weight.hdf5')`

### Credits and references
- https://www.analyticsvidhya.com/blog/2020/01/how-to-perform-automatic-music-generation/
- https://medium.com/@alexissa122/generating-original-classical-music-with-an-lstm-neural-network-and-attention-abf03f9ddcb4
- https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
- http://colah.github.io/posts/2015-08-Understanding-LSTMs/
