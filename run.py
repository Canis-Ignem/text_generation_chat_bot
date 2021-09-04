import data_handler as dh
from torch import nn
import torch
import models
import vocab

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 512
encoder_n_layers = 4
decoder_n_layers = 4
dropout = 0.2
batch_size = 128



USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

voc = vocab.Voc("Testing")

checkpoint = torch.load("models/9500_checkpoint.tar")
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
embedding_sd = checkpoint['embedding']
voc.__dict__ = checkpoint['voc_dict']


embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)

encoder = models.EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = models.LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)
encoder.eval()
decoder.eval()



def evaluate(encoder, decoder, searcher, voc, sentence, max_length=dh.MAX_LENGTH):

    indexes_batch = [dh.indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = dh.normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

# Initialize search module
searcher = models.GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)
