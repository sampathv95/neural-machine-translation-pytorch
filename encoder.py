import torch

class Encoder(torch.nn.Module):


    def __init__(self, n_layers, input_dim, embedding_dim, hidden_dim, **kwargs):
        # initializing the inherited module class
        super().__init__()
        
        # initializing a new word embedding layer
        self.embedding_layer = torch.nn.Embedding(input_dim, embedding_dim)

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

        self.dropout_ratio = None
        if kwargs.get('dropout') is not None:
            self.dropout_ratio = kwargs.get('dropout')
            # dropout layer
            self.dropout_layer = torch.nn.Dropout(self.dropout_ratio)
        

    def forward(self, inputs):
        # inputs = (batch_size, src_len)

        embedding_out = self.embedding_layer(inputs)
        if self.dropout_ratio is not None:
            embedding_out = self.dropout(embedding_out)
        # embeddings_out = (batch_size, embedding_dim)

        outputs, (hidden_states, cell_states) = self.lstm(embedding_out)
        # outputs = (batch_size, src_len, hidden_dim)
        # hidden_states = (batch_size, n_layers, hidden_dim)
        # cell_states = (batch_size, b_layers, hidden_dim)

        # returning hidden and cell states for initializing the decoder
        return hidden_states, cell_states

if __name__ == '__main__':
    # testing
    encoder = Encoder(n_layers=2, input_dim=1000, embedding_dim=256, hidden_dim=64, dropout=0.2)
    print(encoder)

        
        


    

        

    
