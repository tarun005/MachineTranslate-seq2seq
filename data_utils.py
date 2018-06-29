import torch , torch.utils.data
import torch.nn as nn , torch.nn.functional as F
from nltk.tokenize import word_tokenize
import numpy as np
import os , sys
import re
import itertools

import unicodedata

def preprocess_string(list_of_sent):
	ret_list = []

	if not isinstance(list_of_sent , list):
		list_of_sent = [list_of_sent]

	for s in list_of_sent:
		s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
		s = s.lower() ## Lower case
		s = re.sub(r'[!.,\[\]()?:;{}]' , ' ' , s) ## Remove Punctuations
		s = re.sub(r'[\d]+' , '<d>' , s) ## Replace digits
		ret_list.append(s)

	return ret_list

class prepare_data():

	def __init__(self , config , debug=False):

		self.build_corpus(config , debug)
		self.build_vocab(config)

	def build_corpus(self , config , debug):
		## Make a corpus with source-target languages as tuples
		if isinstance(config.file_name , list):
			with open(os.path.join(config.root_dir , config.file_name[0]) , 'r' , encoding='utf-8') as fh:
				corpus_source = [preprocess_string(line.strip())[0] for line in fh]

			with open(os.path.join(config.root_dir , config.file_name[1]) , 'r' , encoding='utf-8') as fh:
				corpus_target = [preprocess_string(line.strip())[0] for line in fh]

			corpus = [tuple([corpus_source[i] , corpus_target[i]]) for i in range(len(corpus_source))]

		else:
			with open(os.path.join(config.root_dir , config.file_name), 'r' , encoding='utf-8') as fh:
			    corpus = [tuple(preprocess_string(line.strip().split(config.delimiter))) for line in fh]

		corpus = corpus[:700000]
		if debug:
			config.n_epochs = 2
			corpus = corpus[:4000]

		total_corpus_size = len(corpus)
		mask = [1 if len(word_tokenize(sample[0])) < config.max_source_len else 0 for sample in corpus]
		corpus = list(itertools.compress(corpus , mask))

		print('Compressed corpus from {} samples to {}'.format(total_corpus_size , len(corpus)))

		np.random.shuffle(corpus)
		total_corpus_size = len(corpus)
		split = int(total_corpus_size*config.train_size)

		data = {'train':corpus[:split] , 'val':corpus[split:]}
		self.data = {phase:sorted(data[phase] , key = lambda i: len(word_tokenize(i[0]))) for phase in ['train','val']}
		self.data_loader = {phase: torch.utils.data.DataLoader(data[phase], shuffle=False, 
									batch_size=config.batch_size, num_workers=2) for phase in ['train' , 'val']}


	def build_vocab(self , config):
		## Build a vocabulary set of source-target languages

		word_freq = {config.source:{} , config.target:{}}
  
		train_corpus = self.data['train']
		
		for sample_id in train_corpus:
			for i , lang in enumerate([config.source , config.target]):
			    for word in word_tokenize(sample_id[i]):
			        word_freq[lang][word] = word_freq[lang].get(word , 0) + 1

		self.vocab = {config.source:{config.unknown_tok:0 , config.end_tok:1} , 
						config.target:{config.unknown_tok:0 , config.end_tok:1 , config.start_tok:2}}

		for lang in [config.source, config.target]:
		    for word , freq in word_freq[lang].items():
		        if freq >= config.min_word_freq:
		            self.vocab[lang][word] = len(self.vocab[lang]) 

		self.vocab_size = {config.source:len(self.vocab[config.source]) , config.target:len(self.vocab[config.target])}

	def seq_to_idx(self, config, seq ,source=True, sort_idx=None):
	    
	    output_idx = []
	    seq_len = []
	    idx_dict = self.vocab[config.source if source else config.target]
	    
	    if source:
	        sort_idx = sorted(np.arange(len(seq)) , reverse=True, key = lambda i: len(word_tokenize(seq[i])))
	        seq = [seq[i] for i in sort_idx]   
	    elif sort_idx is None:  
	        raise ValueError("You should provide an ordering sequence between target batch sequences.")
	    else:
	        seq = [seq[i] for i in sort_idx]
	    
	    max_seq_len = max([len(word_tokenize(s)) for s in seq])
	    
	    for curr_seq in seq:
	        words = word_tokenize(curr_seq)
	        seq_len.append(min(len(words) , max_seq_len) +1)
	        words = words[:max_seq_len] + [config.end_tok] + [config.unknown_tok]*max(0 , max_seq_len - len(words))      
	        output_idx.append([idx_dict.get(word , idx_dict[config.unknown_tok]) for word in words])
	        # print("##########")
	        # for word in words:
	        # 	if word not in idx_dict:
	        # 		print(word , end = '')
	        # print()
	        # print("##########")
	    return torch.tensor(output_idx , dtype=torch.long) , seq_len, sort_idx

if __name__ == "__main__":
	# f = prepare_data(0)
    raise NotImplementedError("Sub modules are not callable")
