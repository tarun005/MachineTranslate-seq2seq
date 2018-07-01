# MT_seq2seq
PyTorch implementation of basic sequence to sequence (seq2seq) encoder-decoder architecture for machine translation.

## How to run
Use the command
`python main.py <root_location> <filename(s)> <save_path> -attn=dot -debug=True`
from the home folder. 
* `<root_location>` is the place consisting of the datafiles
* `<filename(s)>` is the _name_ of the parallel corpus file(s). It can be
  * a single value corresponding to the file which contains the source-target pair in tab separated format
  * or two files corresponding to the source tokens and target tokens respectively, aligned by each line. 
* `<save_path>` is the location to save the trained model.
* `-attn` (optional, default='dot') specifies the attention model. 
* `True` or `False` value for `-debug` toggles the debug mode.  

## Architecture Details
* The architecture follows a similar pattern to [Bahadanau et. al.](https://arxiv.org/abs/1409.0473). A deep GRU model is used as an encoder, and another deep GRU model is used to decode and output a meaningful translation. 
* Batch processing is possible, and all sentences in a batch are pre processed to be of almost equal length. 
* _Masked cross entropy loss_ is implemented at the decoder, where any tokens padded for ease of batch processing are excluded in computing the loss, making use of pytorch's [pack_padded_sequence and pad_packed_sequence](https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e).

## Attention module
The different kinds of attention model implemented are 
* __None__ : No attention module would be used during decoding.
* __dot__ : ![dot](https://latex.codecogs.com/svg.latex?%5Cinline%20s_i%20%5Cpropto%20h_i%5ET%20%5Ccdot%20h_s)
* __linear__ : ![linear](https://latex.codecogs.com/svg.latex?%5Cinline%20s_i%20%5Cpropto%20v%5ET%20%5Ccdot%20%5Ctext%7BRelu%7D%28W%5Bh_i%20%3B%20h_d%5D%20&plus;%20b%29)
* __bilinear__ : ![bilinear](https://latex.codecogs.com/svg.latex?%5Cinline%20s_i%20%5Cpropto%20h_i%5ET%20%5Ccdot%20W%20%5Ccdot%20h_s)

## Parameter tuning
Most of the hyper parameters have been provided with suitable default values in [main.py](main.py). Some hyperparameters I think are worth changing for better performance are listed below.
* __teacher_forcing_ratio__ : If this value is 1, then the outputs fed to the next step of RNN are the ground truth words, called _teacher forcing_. If 0, then the predictions of the previous method are fed as inputs, called _scheduled sampling_. This value corresponds to probability of using teacher forcing at each iteration. 
* __n_layers__ : Deeper GRUs are known to give better results compared to shallower GRU. [Sutskever et. al.](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) use 4 layers at encoder and decoder, while [Google's NMT](https://arxiv.org/abs/1609.08144) uses 8 on each side. 
* __dropout__ : Increasing the value of dropout reduces overfitting

## Sample results for ENGLISH-HINDI translation.

The network is run on the [OpenSubtitles](http://stp.lingfil.uu.se/~joerg/published/ranlp-V.pdf) dataset for en-hi translation, which consists of over 90,000 sentence pairs.

### Sample translations:

 
