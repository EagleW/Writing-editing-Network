import torch.nn as nn
import torch.nn.functional as F


class FbSeq2seq(nn.Module):

    def __init__(self, encoder_title, encoder, decoder, decode_function=F.log_softmax):
        super(FbSeq2seq, self).__init__()
        self.encoder_title = encoder_title
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, prev_generated_seq=None, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder_title(input_variable, input_lengths)
        if prev_generated_seq is None:
            pg_encoder_states = None
            pg_encoder_hidden = None
        else:
            pg_encoder_states, pg_encoder_hidden = self.encoder(prev_generated_seq)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              pg_encoder_states=pg_encoder_states,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
