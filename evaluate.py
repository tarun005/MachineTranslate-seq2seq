import torch
from nltk.tokenize import word_tokenize
from nltk.translate import bleu_score
from data_utils import preprocess_string

def give_idx_to_phrase(config, vocab, inp_idx, target=True):

    """
    Given a torch tensor of size batch_size * seq_len, it gives out <batch_len> number of sentences corresponding to the 
    indices. Used to view results as well as calculate BLEU scores. 
    """

    if target:
        reverse_vocab = {i:word for word,i in vocab[config.target].items()}
    else:
        reverse_vocab = {i:word for word,i in vocab[config.source].items()}
    
    inp_idx = inp_idx.cpu().numpy()
    phrase = []
    
    for sample in range(inp_idx.shape[0]):
        word_list = []
        sen_token = inp_idx[sample,:]

        for idx in sen_token:
            token = reverse_vocab[idx]
            if token == config.end_tok:
                break;
            else:
                word_list.append(token)
            
        phrase.append(' '.join(word_list))
     
    return phrase ## Returns list of strings

def generate_translation(input_string, config, encoder, decoder, vocab):

    input_string = preprocess_string(input_string)

    with torch.no_grad():
        input_idx = [vocab[config.source].get(word , vocab[config.source][config.unknown_tok])
        													 for word in word_tokenize(input_string)]
        input_idx = input_idx + [vocab[config.source][config.end_tok]]

        input_idx_tensor = torch.tensor(input_idx , dtype=torch.long).to(config.device)
        input_idx_tensor = input_idx_tensor.view([1,-1])
        
        encoding_output , encoded_embedding = encoder(input_idx_tensor , [len(input_idx)])
        encoded_embedding = encoded_embedding[-config.n_layers_decoding:]
        
        start_token = torch.tensor(vocab[config.target][config.start_tok]).long().to(config.device).view(-1,1)
        
        reverse_vocab = {phase:{i:word for word,i in vocab[phase].items()} for phase in [config.source , config.target]}
        
        next_token = start_token
        next_hidden = encoded_embedding
        stop_condition = False
        output_tokens = []
        n_iters = 0
        
        while stop_condition is False:
            output_prob , next_hidden = decoder(encoding_output, next_token , next_hidden)
            next_token = torch.argmax(output_prob , dim=2)
            word_token = next_token.item()

            if n_iters == config.max_target_len or reverse_vocab[config.target][word_token] == config.end_tok:
                stop_condition = True

            output_tokens.append(reverse_vocab[config.target][word_token])    
            n_iters += 1
        
    output_string = ' '.join(output_tokens[:-1]) ## Ignore the <eos> string
    return output_string

def BLEU_score(config, gt_caption, sample_caption):
    """
    gt_caption: string or list of string, ground-truth caption
    sample_caption: string or list of strings, model's predicted caption
    Returns batch sum of unigram BLEU score.
    """
    gt = preprocess_string(gt)
    sample = preprocess_string(sample)
    bleu_scores = 0

    for gt , sample in zip(gt_caption , sample_caption):

        reference = [x for x in word_tokenize(gt) 
                     if (config.end_tok not in x and config.start_tok not in x and config.unknown_tok not in x)]
        hypothesis = [x for x in word_tokenize(sample) 
                      if (config.end_tok not in x and config.start_tok not in x and config.unknown_tok not in x)]

        BLEUscore = bleu_score.sentence_bleu([reference], hypothesis, weights = [1])
        bleu_scores += BLEUscore

    return bleu_scores

if __name__ == "__main__":
    raise NotImplementedError("Sub modules are not callable")