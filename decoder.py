import torch

class Decoder(torch.nn.Module):


    def __init__(self, n_layers, output_dim, embedding_dim, hidden_dim, **kwargs):
        # initializing the inherited module class
        super().__init__()
        self.output_dim = output_dim

        # initializing a new word embedding layer
        self.embedding_layer = torch.nn.Embedding(output_dim, embedding_dim)

        # relace the new embedding layer with pre-trained layer if present
        # in case of using pre-trained word embeddings
        if kwargs.get('embedding_layer') is not None:
            embedding_dim = kwargs.get('embedding_layer_dim')
            self.embedding_layer = kwargs.get('embedding_layer')
            # making the weights unaffected by back propagation
            self.embedding_layer.weight.requires_grad = False

        # initializing new LSTM layer
        self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                                    num_layers=n_layers, batch_first=True)
        
        self.linear_out = torch.nn.Linear(in_features=hidden_dim, out_features=output_dim)

        self.dropout_ratio = None
        if kwargs.get('dropout') is not None:
            self.dropout_ratio = kwargs.get('dropout')
            # dropout layer
            self.dropout_layer = torch.nn.Dropout(self.dropout_ratio)

        
    def forward(self, inputs, hidden_states, cell_states):
        # inputs = (batch_size, trg_len)

        embedding_out = self.embedding_layer(inputs)
        if self.dropout_ratio is not None:
            embedding_out = self.dropout_layer(embedding_out)
        # embedding_out = (batch_size, embedding_dim)

        outputs, (hidden_states, cell_states) = self.lstm(embedding_out, (hidden_states, cell_states))
        # outputs = (batch_size, src_len, hidden_dim)
        # hidden_states = (batch_size, n_layers, hidden_dim)
        # cell_states = (batch_size, b_layers, hidden_dim)

        # converting each vector of length hidden_dim to length of output_dim (TimeDistributed Layer)
        outputs_numer = torch.zeros(outputs.shape[0], outputs.shape[1], self.output_dim)

        for i in range(outputs.shape[0]):
            outputs_numer[i] = self.linear_out(outputs[i])
        
        return outputs_numer

if __name__ == '__main__':
    dec = Decoder(n_layers=2, embedding_dim=256, output_dim=1000, hidden_dim=64, dropout=0.2)
    dec_out = dec(torch.zeros(128, 10, dtype=torch.long), torch.zeros(2, 128, 64, dtype=torch.float32), torch.zeros(2, 128, 64, dtype=torch.float32))
    print(dec_out)
    print(dec_out.shape)






