import torch , torch.nn as nn
import torch.nn.functional as F

class decoderAttn(nn.Module):
    
    def __init__(self , attn_model, embedding_size , vocab_size, hidden_size, n_layers=1):
        """
        Attn Model should be one of dot or linear.
        """
        super(decoderAttn , self).__init__()

        ## One of 'dot' , 'bilinear' , or 'linear'. See https://youtu.be/IxQtK2SjWWM?t=55m49s
        self.attn_model = attn_model 

        self.embedding = nn.Embedding(vocab_size , embedding_size)

        self.rnn = nn.GRU(embedding_size + hidden_size , hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*2 , vocab_size)

        self.attention = attention(self.attn_model, hidden_size)
        
    def forward(self, encoding_output, input_token_v , hidden_state):
        
        ## input_token_v is a tensor of shape [batch_size]
        ## hidden_state is of shape [n_layers_decoding, batch_size, hidden_size]
        
        context_vector = self.attention(encoding_output , hidden_state[-1]) ## [batch_size , hidden_size]
        input_vector = F.relu(self.embedding(input_token_v)) ## [batch_size , embedding_size]
        rnn_input = torch.cat([input_vector , context_vector] , dim=1).unsqueeze(1) ## [batch_size, seq_len=1 ,embedding_size]

        op, next_hidden_states = self.rnn(rnn_input , hidden_state)
        assert(op.size(1) == 1)
        op = op.squeeze(1) ## [batch_size , hidden_size]
        fc_output = torch.cat([op , context_vector] , dim=1)

        scores = self.fc(fc_output) ## [batch_size , output_vocab_size]

        
        return scores , next_hidden_states 


class attention(nn.Module):

    def __init__(self , attn_model, hidden_size):

        super(attention , self).__init__()

        self.attn_model = attn_model

        if attn_model == 'linear':
            self.attn = nn.Linear(2*hidden_size , hidden_size)
            self.v = nn.Parameter(torch.rand(1,hidden_size))
            self.v.data.normal_(0 , 0.002) ## Small random values
        elif attn_model == 'bilinear':
            raise NotImplementedError

    def forward(self, encoder_outputs, hidden_state):

        """
        encoder_outputs : [batch_size , seq_len , hidden_size]
        hidden_state : [batch_size , hidden_size]
        """
        
        attention_energies_unnormalized = self.score(hidden_state, encoder_outputs) #[batch_size , seq_len]
        attention_weights = F.softmax(attention_energies_unnormalized , dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1) , encoder_outputs).squeeze(1) ## [batch_size , hidden_size]

        return context_vector

    def score(self, hidden_state, encoder_outputs):

        """
        hidden_state : [batch_size , hidden_size]
        encoder_outputs : [batch_size , seq_len , hidden_size]
        """

        if self.attn_model == 'dot':
            attention_energies_unnormalized = torch.bmm(encoder_outputs , hidden_state.unsqueeze(-1)).squeeze(-1)

        elif self.attn_model == 'linear':
            hidden_state = hidden_state.unsqueeze(1).repeat(1 , encoder_outputs.size(1) ,1)
            fc_input = torch.cat([hidden_state , encoder_outputs] , dim=2) ## batch_size , ip_seq_len , 2*hidden_size
            interm_op = F.relu(self.attn(fc_input)) ## batch_size , ip_seq_len , hidden_size

            self.v = self.v.repeat(encoder_outputs.shape(0) , 1).unsqueeze(-1) ## batch_size , hidden_size ,1
            attention_energies_unnormalized = torch.bmm(interm_op , self.v).squeeze(-1) ## batch_size , ip_seq_len

        elif self.attn_model == 'bilinear':
            pass

        else:
            raise NotImplementedError

        return attention_energies_unnormalized


if __name__ == "__main__":
    raise NotImplementedError("Sub modules are not callable")
