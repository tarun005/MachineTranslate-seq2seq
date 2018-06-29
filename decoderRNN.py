import torch , torch.nn as nn
import torch.nn.functional as F

class decoder_cell(nn.Module):
    
    def __init__(self , embedding_size , vocab_size, hidden_size, n_layers=1):
        self.n_layers = n_layers

        super(decoder_cell , self).__init__()
        self.embedding = nn.Embedding(vocab_size , embedding_size)
        self.rnn_cell = nn.GRUCell(embedding_size , hidden_size)
        self.fc = nn.Linear(hidden_size , vocab_size)

        self.dest_rnn = [self.rnn_cell]*self.n_layers 
        
    def forward(self ,encoding_output, input_token_v , hidden_state):
        
        ## input_token_v is a tensor of shape [batch_size , embed_dim]
        ## hidden_state is a list of states or a context vector

        if not isinstance(hidden_state , list): 
            zero_hidden_state = torch.zeros_like(hidden_state)
            hidden_state = [hidden_state] + [zero_hidden_state]*(self.n_layers - 1)

        assert(len(self.dest_rnn) == len(hidden_state))
        
        input_vector = F.relu(self.embedding(input_token_v))
        next_hidden_states = []

        ip_v = input_vector
        for layer_id in range(self.n_layers):
            op = self.dest_rnn[layer_id](ip_v , hidden_state[layer_id])
            next_hidden_states.append(op)
            ip_v = op

        seq_output = self.fc(op)
        
        return seq_output , next_hidden_states 

if __name__ == "__main__":
    raise NotImplementedError("Sub modules are not callable")
