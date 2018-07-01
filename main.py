import torch , torch.utils.data
import torch.nn as nn , torch.nn.functional as F
import numpy as np
import sys,time,os
import argparse

from data_utils import prepare_data
from encoderRNN import encoder_RNN
from decoderAttn_RNN import decoderAttn
from train import train
from evaluate import generate_translation, BLEU_score

class Config():

	parser = argparse.ArgumentParser(description="Machine translation using seq2seq deep LSTM architecture.")
	parser.add_argument('root' , help='Path of the root dir containing the dataset files' , default='.' , nargs='?')
	parser.add_argument('filenames' , help='One file with tab separated tokens or two files with source and target tokens' ,nargs='+')
	parser.add_argument('save_path' , help='Filename for saving the model.')
	parser.add_argument('-attn' , help='Type of attention model', default='dot')
	parser.add_argument('-debug' , help='True for debug mode', default=False, type=bool)

	args = parser.parse_args()
	root_dir = args.root
	file_name = args.filenames if len(args.filenames) == 2 else args.filenames[0] 
	save_path = args.save_path
	attn_model = args.attn
	debug = args.debug

	start_tok = '<start>'
	end_tok = '<end>'
	unknown_tok = '<unk>'
	source = 'lang1'
	target = 'lang2'
	delimiter = '\t'

	n_layers_encoding=2
	n_layers_decoding=2
	bidirectional=True
	teacher_forcing_ratio = 0.5
	batch_size = 128
	lr = 5e-4
	dropout = 0.4
	embedding_size=256
	hidden_size=256
	max_source_len = 15
	max_target_len = 20
	min_word_freq = 1 ## Common for both the vocab
	train_size = 0.98 ## Fraction for train data
	n_epochs = 40

	## Use GPU
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():

	config = Config()
	data = prepare_data(config , debug=config.debug)
	data.data_loader = {phase: torch.utils.data.DataLoader(data.data[phase], shuffle=False, 
									batch_size=config.batch_size, num_workers=2) for phase in ['train' , 'val']}


	encoder_model = encoder_RNN(config.embedding_size, data.vocab_size[config.source], config.hidden_size , 
		n_layers=config.n_layers_encoding , bidirectional=config.bidirectional, dropout=config.dropout).to(config.device)

	decoder_model = decoderAttn(config.attn_model , config.embedding_size , data.vocab_size[config.target], config.hidden_size, 
	                             n_layers=config.n_layers_decoding, dropout=config.dropout).to(config.device)

	loss_criterion = nn.CrossEntropyLoss(reduce=False)
	encoder_optimizer = torch.optim.Adam(encoder_model.parameters() , lr=config.lr)
	decoder_optimizer = torch.optim.Adam(decoder_model.parameters() , lr=config.lr)

	start = time.time()
	encoder_model , decoder_model , loss_curve = train(data, config, encoder_model, decoder_model, 
											encoder_optimizer, decoder_optimizer,loss_criterion,n_epochs=config.n_epochs)

	print('Training Completed. Took {} seconds'.format(time.time()-start))


	## Evaluate
	print("######## VALIDATION #########")
	for i in np.random.randint(0, len(data.data['val']), 5):
	    inp_seq = data.data['val'][i][0] + config.end_tok
	    print(data.data['val'][i][0])
	    print(data.data['val'][i][1])
	    gen_sen = generate_translation(inp_seq ,config, encoder_model , decoder_model, data.vocab)
	    print(gen_sen)
	    print(BLEU_score(config , data.data['val'][i][1], gen_sen))
	    print()

	print("######## TRAIN #########")
	for i in np.random.randint(0, len(data.data['train']), 5):
	    inp_seq = data.data['train'][i][0] + config.end_tok
	    print(data.data['train'][i][0])
	    print(data.data['train'][i][1])
	    gen_sen = generate_translation(inp_seq ,config, encoder_model , decoder_model, data.vocab)
	    print(gen_sen)
	    print(BLEU_score(config , data.data['train'][i][1], gen_sen))
	    print()


	return config , data

if __name__ == "__main__":

	try:
		config , data = main()
	except KeyboardInterrupt: ## Often, the code takes too long to run, but life is too short.
		print() ## Message
		sys.exit(0)
