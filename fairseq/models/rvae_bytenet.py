import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from fairseq import utils
from fairseq.models import FairseqEncoder
from fairseq.models import FairseqDecoder
from fairseq.models import FairseqModel, register_model
from fairseq.models import register_model_architecture


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-8):
        super(LayerNorm, self).__init__()

        self.eps = eps

        self.sigma = nn.Parameter(torch.ones(size))
        self.mu = nn.Parameter(torch.zeros(size))

    def forward(self, z):

        z = z.transpose(1, 2)

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        out = (z - mu) / (sigma + self.eps)
        out = out * self.sigma.expand_as(out) + self.mu.expand_as(out)

        return out.transpose(1, 2)


class ResBlock(nn.Module):

    def __init__(self, hidden_dim, dilation=1, kernel_size=3):
        super(ResBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.block = nn.Sequential(
            LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),

            LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.ConstantPad1d((ResBlock.same_pad(kernel_size, dilation), 0), 0.),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=kernel_size, dilation=dilation),

            LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=1)
        )

    @staticmethod
    def same_pad(k=1, dil=1):
        p = math.ceil(dil * (k - 1))
        return p

    def forward(self, input):
        return input + self.block(input)


class VAELSTMEncoder(FairseqEncoder):

    def __init__(
        self, args, dictionary, embed_dim=128, hidden_dim=128, dropout=0.1,
    ):
        super().__init__(dictionary)
        self.args = args

        # Our encoder will embed the inputs before feeding them to the LSTM.
        self.embed_tokens = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad(),
        )
        self.dropout = nn.Dropout(p=dropout)

        self.hidden_dim = hidden_dim

        # We'll use a single-layer, unidirectional LSTM for simplicity.
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
        )

        self.context_to_mu = nn.Linear(hidden_dim * 2, hidden_dim)
        self.context_to_logvar = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src_tokens, src_lengths):
        # The inputs to the ``forward()`` function are determined by the
        # Task, and in particular the ``'net_input'`` key in each
        # mini-batch. We discuss Tasks in the next tutorial, but for now just
        # know that *src_tokens* has shape `(batch, src_len)` and *src_lengths*
        # has shape `(batch)`.

        # Note that the source is typically padded on the left. This can be
        # configured by adding the `--left-pad-source "False"` command-line
        # argument, but here we'll make the Encoder handle either kind of
        # padding by converting everything to be right-padded.
        if self.args.left_pad_source:
            # Convert left-padding to right-padding.
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                padding_idx=self.dictionary.pad(),
                left_to_right=True
            )

        bsz, seqlen = src_tokens.size()

        # Embed the source.
        x = self.embed_tokens(src_tokens)

        # Apply dropout.
        x = self.dropout(x)

        # Pack the sequence into a PackedSequence object to feed to the LSTM.
        x = nn.utils.rnn.pack_padded_sequence(x, src_lengths, batch_first=True)

        # Get the output from the LSTM.
        _outputs, (_final_hidden, _final_cell) = self.lstm(x)

        x, _ = nn.utils.rnn.pad_packed_sequence(_outputs, padding_value=0)

        assert list(x.size()) == [seqlen, bsz, 2*self.hidden_dim]

        final_hidden = torch.mean(x, dim=0)

        assert list(final_hidden.size()) == [bsz, 2*self.hidden_dim]

        mu = self.context_to_mu(final_hidden)
        logvar = self.context_to_logvar(final_hidden)

        std = torch.exp(0.5 * logvar)
        z = torch.randn(mu.size())
        if torch.cuda.is_available():
            z = z.cuda()
        z = z * std + mu
        # Return the Encoder's output. This can be any object and will be
        # passed directly to the Decoder.
        return {
            # this will have shape `(bsz, hidden_dim)`
            'final_hidden': z,
            'logvar': logvar,
            'mu': mu
        }

    # Encoders are required to implement this method so that we can rearrange
    # the order of the batch elements during inference (e.g., beam search).
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        final_hidden = encoder_out['final_hidden']

        return {
            'final_hidden': final_hidden.index_select(0, new_order),
            'logvar': encoder_out['logvar'],
            'mu': encoder_out['mu'],
        }


class ByteNetDecoder(FairseqDecoder):

    def __init__(
            self, dictionary, encoder_hidden_dim=128, embed_dim=128,
            dropout=0.1, dilations='1,2,4,8'
    ):
        super().__init__(dictionary)

        self.decoder_dilations = [int(x) for x in dilations.split(',')]

        self.dropout = nn.Dropout(p=dropout)

        self.embed_tokens = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad(),
        )

        self.hidden_dim = encoder_hidden_dim + embed_dim

        self.layers = nn.ModuleList([])

        self.layers.extend([
            ResBlock(self.hidden_dim, dilation, kernel_size=2)
            for dilation in self.decoder_dilations
        ])

        self.output_projection = nn.Linear(self.hidden_dim, len(dictionary))

    def forward(self, prev_output_tokens, encoder_out):

        final_encoder_hidden = encoder_out['final_hidden']

        bsz, tgt_len = prev_output_tokens.size()

        # Embed the source.
        x = self.embed_tokens(prev_output_tokens)

        # Apply dropout.
        x = self.dropout(x)

        x = torch.cat(
            [x, final_encoder_hidden.unsqueeze(1).expand(bsz, tgt_len, -1)],
            dim=2,
        )

        x = x.transpose(1, 2).transpose(0, 2)

        for layer in self.layers:
            x = layer(x)

        x = x.transpose(2, 0).transpose(2, 1)

        x = x.transpose(0, 1)

        x = self.output_projection(x)

        return x, None, encoder_out['logvar'], encoder_out['mu']


@register_model('bytenet_rvae')
class ByteNetRVAE(FairseqModel):
    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add some new command-line arguments to configure dropout
        # and the dimensionality of the embeddings and hidden states.
        parser.add_argument(
            '--encoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the encoder embeddings',
        )
        parser.add_argument(
            '--encoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the encoder hidden state',
        )
        parser.add_argument(
            '--encoder-dropout', type=float, default=0.1,
            help='encoder dropout probability',
        )
        parser.add_argument(
            '--decoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the decoder embeddings',
        )
        parser.add_argument(
            '--decoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the decoder hidden state',
        )
        parser.add_argument(
            '--decoder-dropout', type=float, default=0.1,
            help='decoder dropout probability',
        )
        parser.add_argument(
            '--dilations', type=str, default='1,2,4,8',
            help='dilations for masked convolutions'
        )


    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a SimpleLSTMModel instance.

        # Initialize our Encoder and Decoder.
        encoder = VAELSTMEncoder(
            args=args,
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_dim=args.encoder_hidden_dim,
            dropout=args.encoder_dropout,
        )
        decoder = ByteNetDecoder(
            dictionary=task.target_dictionary,
            encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            dropout=args.decoder_dropout,
            dilations=args.dilations,
        )
        model = ByteNetRVAE(encoder, decoder)

        # Print the model architecture.
        print(model)

        return model

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out


@register_model_architecture('bytenet_rvae', 'bytenet_rvae')
def bytenet_rvae(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)

