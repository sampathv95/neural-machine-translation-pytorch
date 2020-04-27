import torch
import numpy as np

def generate_word_embeddings(vocab_dict, pre_trained_path, embedding_dim, start_token, end_token, **kwargs):

    # reading the embeddings file
    with open(pre_trained_path, 'r') as embedding_file:
        glove_embeddings = embedding_file.readlines()
    
    embeddings_dict = {}
    for line in glove_embeddings:
        word, embedding = line.split()[0], line.split()[1:]
        embeddings_dict[word] = np.asarray(embedding, np.float32)

    embeddings_list = []
    for word, id in vocab_dict.items():
        if word == start_token:
            embeddings_list.append(np.array([0.1]*embedding_dim))
        elif word == end_token:
            embeddings_list.append(np.array([0.2]*embedding_dim))
        else:
            embeddings_list.append(embeddings_dict.get(word, np.random.normal(size=(embedding_dim,))))

    embedding_tensor = torch.from_numpy(np.array(embeddings_list))
    
    if kwargs.get('save'):
        torch.save(embedding_tensor, kwargs.get('path'))
    
    return embedding_tensor

    
