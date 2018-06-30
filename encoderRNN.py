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
        
    def forward(self, input_wv, seq_len):
        
        ip = F.dropout(self.embedding(input_wv) , p=self.dropout)
        packed_ip_seq = nn.utils.rnn.pack_padded_sequence(ip , seq_len , batch_first=True)
        rnn_output, last_hidden = self.source_rnn(packed_ip_seq)
        encoding_output , op_seq_len = nn.utils.rnn.pad_packed_sequence(rnn_output , batch_first=True)

        # if self.bidirectional:
        #     context_vector = torch.cat([last_hidden[-2,:,:] , last_hidden[-1,:,:]] , dim=1)
        # else:
        #     context_vector = last_hidden[-1,:,:]
        
        return encoding_output , last_hidden

if __name__ == "__main__":
    
    raise NotImplementedError("Sub modules are not callable")