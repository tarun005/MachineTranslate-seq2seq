import torch , torch.nn as nn
import torch.nn.functional as F

class encoder_RNN(nn.Module):
    
    def __init__(self, embedding_size, vocab_size, hidden_size, n_layers=1, bidirectional=False, dropout=0):

        super(encoder_RNN , self).__init__()
        self.embedding = nn.Embedding(vocab_size , embedding_size)
        self.source_rnn = nn.GRU(embedding_size , hidden_size , dropout=dropout,
                                   num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.hidden_size = hidden_size
        
    def forward(self, input_wv, seq_len):
        
        ip = F.dropout(self.embedding(input_wv) , p=self.dropout)
        packed_ip_seq = nn.utils.rnn.pack_padded_sequence(ip , seq_len , batch_first=True)

        ## https://pytorch.org/docs/stable/nn.html#torch.nn.GRU
        rnn_output, last_hidden = self.source_rnn(packed_ip_seq)
        encoding_output , _ = nn.utils.rnn.pad_packed_sequence(rnn_output , batch_first=True) 

        if self.bidirectional:
            ## Add contributions from both directions. 
            ## Can also try torch.cat, but decoder hidden size should be doubled.
            encoding_output = torch.sum(encoding_output[:,:,:self.hidden_size] , encoding_output[:,:,self.hidden_size:])
            last_hidden = torch.sum(last_hidden[0:2:,:,:] , last_hidden[1:2:,:,:])

        return encoding_output , last_hidden

if __name__ == "__main__":
    
    raise NotImplementedError("Sub modules are not callable")