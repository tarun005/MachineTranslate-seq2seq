# MT_seq2seq
PyTorch implementation of basic sequence to sequence (seq2seq) encoder-decoder architecture for machine translation.

## How to run
Use the command
`python main.py <root_location> <filename(s)> <save_path> -attn=dot -debug=True`
from the home folder. 
* `<root_location>` is the place consisting of the datafiles
* `<filename(s)>` is the location of the parallel corpus file(s). It can be
  * a single value corresponding to the file which contains the source-target tokens in tab separated format
  * or two files corresponding to the source tokens and target tokens respectively, _aligned_. 
* `<save_path>` is the location to save the trained model.
* `-attn` (optional, default='dot') specifies the attention model. 
* `True` or `False` value for `-debug` toggles between the debug mode.  

## Architecture Details
* The architecture follows a similar pattern to [Bahadanau et. al.](https://arxiv.org/abs/1409.0473). A deep GRU model is used as an encoder, and another deep GRU model is used to decode and output a meaningful translation. 
* Batch processing is possible, making use of [pack_padded_sequence and pad_packed_sequence](https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e), and all sentences in a batch are pre processed to be of almost equal length. 

## Attention module
The different kinds of attention model implemented are 
* __None__ : No attention module would be used during decoding.
* __dot__ : ![dot](https://latex.codecogs.com/svg.latex?%5Cinline%20s_i%20%5Cpropto%20h_i%5ET%20%5Ccdot%20h_s)
* __linear__ : ![linear](https://latex.codecogs.com/svg.latex?%5Cinline%20s_i%20%5Cpropto%20v%5ET%20%5Ccdot%20%5Ctext%7BRelu%7D%28W%5Bh_i%20%3B%20h_d%5D%20&plus;%20b%29)
* __bilinear__ : ![bilinear](https://latex.codecogs.com/svg.latex?%5Cinline%20s_i%20%5Cpropto%20h_i%5ET%20%5Ccdot%20W%20%5Ccdot%20h_s)

 
