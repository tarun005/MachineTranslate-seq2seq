import numpy as np
import torch
import sys
import random
import matplotlib.pyplot as plt
from evaluate import give_idx_to_phrase , generate_translation, BLEU_score
from tqdm import tqdm


def train(data, config, encoder , decoder , encoder_optimizer, decoder_optimizer , loss_criterion, n_epochs=7):
    
	best_model_wts = (encoder.state_dict() , decoder.state_dict())
	best_loss = np.inf
	teacher_forcing_ratio = config.teacher_forcing_ratio
	last_epoch_best_loss = np.inf

	if config.train_size!=1:
		phases = ['train' , 'val']
		eval_phase = 'val'
	else:
		phases = ['train']
		eval_phase = 'train'

	total_loss = {phase:[] for phase in phases}

    
	for epoch_id in range(1 , n_epochs+1):
	    print('Epoch {}/{}'.format(epoch_id , n_epochs ))

	    for phase in phases:
	        
		    running_loss = 0
		    bleu_score = 0
		    total_samples = 0

		    if phase == 'train':
		    	encoder = encoder.train()
		    	decoder = decoder.train()
		    else:
		    	encoder = encoder.eval()
		    	decoder = decoder.eval()

		    for input_text , target_text in tqdm(data.data_loader[phase] , position=2, leave=False):

		        batch_size = len(input_text)
		        total_samples += batch_size
		        
		        ## Reset the gradients.
		        encoder.zero_grad()
		        decoder.zero_grad()
		        
		        input_idx , input_seq_len, sort_idx = data.seq_to_idx(config , input_text)
		        input_idx = torch.tensor(input_idx , dtype=torch.long).to(config.device) ## [batch_size , input_seq_len]

		        output_idx, output_seq_len , _ = data.seq_to_idx(config, target_text , source=False, sort_idx=sort_idx)
		        output_idx = torch.tensor(output_idx , dtype=torch.long).to(config.device)  ## [batch_size , output_seq_len]
		        
		        with torch.set_grad_enabled(phase == 'train'): 
		            ## encoding
		            encoding_output , context_vector = encoder(input_idx , input_seq_len)
		            context_vector = context_vector[-config.n_layers_decoding:]

		            ## decoding. 
		            ## No teacher forcing : Use own predictions to feed into the next cell of RNN.
		            ## Teacher forcing : Use ground truth labels to feed as the next word.

		            seq_output = []
		            mask = []
		            ip_seq = torch.tensor([data.vocab[config.target][config.start_tok]]*batch_size 
		            		, dtype=torch.long).view([-1,1]).to(config.device)

		            next_hidden = context_vector
		            use_teacher_forcing = random.random()<teacher_forcing_ratio
		            predicted_sentence = []

		            for iter_n in range(max(output_seq_len)): ## Go till length of longest sequence

		            	## curr_op : [batch_size, seq_len=1, vocab_size]
		            	## next_hidden : [n_layers_decoding, batch_size, hidden_size]
		                curr_op , next_hidden = decoder(encoding_output, ip_seq , next_hidden)
		                op_seq_pred = torch.argmax(curr_op , dim=2) ## [batch_size , seq_len=1]
		                predicted_sentence.append(op_seq_pred)
		                mask += [1 if i>iter_n else 0 for i in output_seq_len] ## Keep creating masks for valid outputs.

		                ip_seq = output_idx[:,iter_n:iter_n+1].detach() if use_teacher_forcing else op_seq_pred.detach()
		                seq_output.append(curr_op)

		            seq_output = torch.cat(seq_output , dim=1) ## [batch_size, output_seq_len, output_size]
		            predicted_sentence = torch.cat(predicted_sentence , dim=1) ## [batch_size , output_seq_len]
		            
		            mask = torch.tensor(mask , dtype=torch.long).nonzero()
		            op_size = seq_output.size(2) ## Output vocab size
		            loss = loss_criterion(seq_output.view([-1,op_size]) , output_idx.view(-1))[mask].mean()

		            gt_sen = give_idx_to_phrase(config , data.vocab, output_idx)
		            pred_sen = give_idx_to_phrase(config , data.vocab, predicted_sentence)
		            bleu_score += BLEU_score(config, gt_sen , pred_sen)

		            if phase == 'train':
		                loss.backward()
		                ## Clip the gradients to solve exploding gradient problem.
		                ## Section 3.4 , https://arxiv.org/pdf/1409.3215v3.pdf
		                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=10)
		                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=10)
		                encoder_optimizer.step()
		                decoder_optimizer.step()

		            if epoch_id == n_epochs and loss.item() < last_epoch_best_loss and phase == eval_phase:
		            	last_epoch_best_loss = loss.item()
		            	best_last_token = input_idx
		            	best_output_token = output_idx

		            total_loss[phase].append(loss.item())
		            running_loss += loss.item()*batch_size
		            
		            sys.stdout.write('\r{}/{} - {} Loss: {:.4f}'.format(total_samples, len(data.data[phase]) 
		            																			, phase, loss.item()))
		            sys.stdout.flush()

		    epoch_loss =  running_loss/total_samples
		    sys.stdout.write('\r{}/{} - {} Loss: {:.4f}'.format(total_samples, len(data.data[phase]) , phase, epoch_loss))
		    sys.stdout.flush()
		    print()
		    if phase == eval_phase and epoch_loss < best_loss:
		        best_loss = epoch_loss
		        best_model_wts = (encoder.state_dict() , decoder.state_dict())
		        torch.save((encoder, decoder, config, data.vocab) , config.save_path)
		    print('{:.4f}: BLEU score'.format(bleu_score/total_samples))
	    print()
	    
	print('Completed Training....')
	print('Best Validation Loss {}'.format(best_loss))
	encoder.load_state_dict(best_model_wts[0])
	decoder.load_state_dict(best_model_wts[1])

	best_source_sen = give_idx_to_phrase(config , data.vocab, best_last_token, target=False)
	best_target_sen = give_idx_to_phrase(config , data.vocab, best_output_token)
	best_pred_sen = []

	for sen in best_source_sen:
		best_pred_sen.append(generate_translation(sen , config, encoder, decoder, data.vocab))

	print()
	print('#'*10)
	for i in range(min(8 , config.batch_size)):
		print(best_source_sen[i])
		print(best_target_sen[i])
		print(best_pred_sen[i])
		print()
	print('#'*10)

	return encoder , decoder , total_loss


def plot_loss(loss):

	"""
	loss is a dictionary with 'train' and 'val' keys
	"""
	assert(isinstance(loss , dict))
	if 'val' in loss:
		plt.plot(loss['train'], 'k' , linewidth=2.5, label='Train loss')
		plt.plot(loss['val'], 'r' , linewidth=2.5, label='Val loss')
	else:
		plt.plot(loss['train'], 'k' , linewidth=2.5, label='Train loss' )

	plt.legend()
	plt.title("Loss Plot")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.savefig('loss_plot.png' , format='png')



if __name__ == "__main__":
    raise NotImplementedError("Sub modules are not callable")