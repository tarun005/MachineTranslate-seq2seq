import torch , torch.nn as nn
import torch.nn.functional as F
import sys

class attention(nn.Module):

    def __init__(self , attn_model, hidden_size):

        super(attention , self).__init__()

        self.attn_model = attn_model

        if attn_model == 'linear':
            self.attn = nn.Linear(2*hidden_size , hidden_size)
            self.v = nn.Parameter(torch.rand(1,hidden_size))
            self.v.data.normal_(0 , 0.002) ## Small random values
        elif attn_model == 'bilinear':
            self.w = nn.Parameter(torch.rand(hidden_size , hidden_size))
            self.w.data.normal_(0, 0.002) ## Small random values

    def forward(self, encoder_outputs, hidden_state):

        """
        encoder_outputs : [batch_size , seq_len , hidden_size]
        hidden_state : [batch_size , hidden_size]
        """

        if self.attn_model.lower() == 'none':
            return torch.zeros_like(hidden_state).unsqueeze(1) ## Context vector would be the zeros vector, no attention

        attention_energies_unnormalized = self.score(hidden_state, encoder_outputs) #[batch_size , seq_len]
        attention_weights = F.softmax(attention_energies_unnormalized , dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1) , encoder_outputs) ## [batch_size, seq_len=1 , hidden_size]

        return context_vector

    def score(self, hidden_state, encoder_outputs):

        """
        hidden_state : [batch_size , hidden_size]
        encoder_outputs : [batch_size , seq_len , hidden_size]
        """

        if self.attn_model.lower() == 'dot':
            attention_energies_unnormalized = torch.bmm(encoder_outputs , hidden_state.unsqueeze(-1)).squeeze(-1) ## batch_size , ip_seq_len

        elif self.attn_model.lower() == 'linear':
            hidden_state = hidden_state.unsqueeze(1).repeat(1 , encoder_outputs.size(1) ,1)
            fc_input = torch.cat([hidden_state , encoder_outputs] , dim=2) ## batch_size , ip_seq_len , 2*hidden_size
            interm_op = F.tanh(self.attn(fc_input)) ## batch_size , ip_seq_len , hidden_size

            v_mat = self.v.repeat(encoder_outputs.size(0) , 1).unsqueeze(-1) ## batch_size , hidden_size ,1
            attention_energies_unnormalized = torch.bmm(interm_op , v_mat).squeeze(-1) ## batch_size , ip_seq_len

        elif self.attn_model.lower() == 'bilinear':
            w_mat = self.w.unsqueeze(0).repeat(encoder_outputs.size(0) , 1 , 1)
            hidden_encoder = torch.bmm(w_mat , hidden_state.unsqueeze(-1))
            attention_energies_unnormalized = torch.bmm(encoder_outputs , hidden_encoder).squeeze(-1) ## batch_size , ip_seq_len   

        return attention_energies_unnormalized


class decoderAttn(nn.Module):
    
    def __init__(self , attn_model, embedding_size , vocab_size, hidden_size, n_layers=1, dropout=0):
       
        super(decoderAttn , self).__init__()

        ## One of 'none' , 'dot' , 'linear' , or 'bilinear'. See https://youtu.be/IxQtK2SjWWM?t=55m49s
        self.attn_model = attn_model 

        self.embedding = nn.Embedding(vocab_size , embedding_size)

        self.rnn = nn.GRU(embedding_size + hidden_size , hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size*2 , vocab_size)

        self.attention = attention(self.attn_model, hidden_size)
        self.dropout = dropout
        
    def forward(self, encoding_output, input_token_v , hidden_state):
        
        ## input_token_v is a tensor of shape [batch_size,1]
        ## hidden_state is of shape [n_layers_decoding, batch_size, hidden_size]
        
        context_vector = self.attention(encoding_output , hidden_state[-1]) ## [batch_size, seq_len=1, hidden_size]
        input_vector = F.dropout(F.relu(self.embedding(input_token_v)) , p=self.dropout, training=self.training) ## [batch_size, seq_len=1 , embedding_size]
        rnn_input = torch.cat([input_vector , context_vector] , dim=2) ## [batch_size, seq_len=1 ,embedding_size + hidden_size]

        op, next_hidden_states = self.rnn(rnn_input , hidden_state) ## op: [batch_size, seq_len=1, hidden_size]
        fc_output = torch.cat([op , context_vector] , dim=2) ## [batch_size , seq_len=1, hidden_size*2]
        scores = self.fc(fc_output) ## [batch_size ,seq_len=1, output_vocab_size]
        
        return scores , next_hidden_states 


if __name__ == "__main__":
    raise NotImplementedError("Sub modules are not callable")
