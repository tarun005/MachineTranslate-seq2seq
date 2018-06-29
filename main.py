import torch , torch.utils.data
import torch.nn as nn , torch.nn.functional as F
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
import sys,time,os

from data_utils import prepare_data
from encoderRNN import encoder_RNN
from decoderRNN import decoder_cell
from decoderAttn_RNN import decoderAttn
from train import train
from evaluate import give_idx_to_phrase , generate_translation, BLEU_score

class Config():

	root_dir = 'fr-en_europarl'
	file_name = ['europarl-v7.fr-en.fr' , 'europarl-v7.fr-en.en'] ## format: source - target
	source = 'FR'
	target = 'EN'
	save_path = 'best_model_attention.pt'

	start_tok = '<start>'
	end_tok = '<end>'
	unknown_tok = '<unk>'
	delimiter = '#'

	embedding_size=256
	hidden_size=256
	n_layers_encoding=2
	n_layers_decoding=2
	max_source_len = 15
	max_target_len = 20
	min_word_freq = 1 ## Common for both the vocab
	bidirectional=False
	train_size = 0.95 ## Fraction for train data
	teacher_forcing_ratio = 0.5
	n_epochs = 60
	batch_size = 32
	lr = 1e-4
	# gamma = 0.5 ## Put to a value <1 to activate scheduler. 1 to deactivate

	## Use GPU
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():

	config = Config()
	data = prepare_data(config , debug=False)


	encoder_model = encoder_RNN(config.embedding_size, data.vocab_size[config.source], 
		config.hidden_size , n_layers=config.n_layers_encoding , bidirectional=config.bidirectional).to(config.device)

	decoder_hidden_size = 2*config.hidden_size if config.bidirectional else config.hidden_size
	decoder_model = decoderAttn('dot' , config.embedding_size , data.vocab_size[config.target], 
	                             decoder_hidden_size, n_layers=config.n_layers_decoding).to(config.device)

	loss_criterion = nn.CrossEntropyLoss(reduce=False)
	encoder_optimizer = torch.optim.Adam(encoder_model.parameters() , lr=config.lr)
	decoder_optimizer = torch.optim.Adam(decoder_model.parameters() , lr=config.lr*5)
	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=config.gamma)

	start = time.time()
	encoder_model , decoder_model , loss_curve = train(data, config, encoder_model, decoder_model, 
											encoder_optimizer, decoder_optimizer,loss_criterion,n_epochs=config.n_epochs)

	print('Training Completed. Took {} seconds'.format(time.time()-start))


	for i in np.random.randint(0, len(data.data['train']), 10):
	    inp_seq = data.data['train'][i][0] + ' eos'
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
		raise
