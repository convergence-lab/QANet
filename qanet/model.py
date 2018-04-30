import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Block
from mxnet.gluon.loss import Loss

class InnerEncoderBlock(Block):
    def __init__(self, num_filters, **kwargs):
        super(InnerEncoderBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.layer_norm = nn.LayerNorm()
            self.conv = nn.Conv2D(num_filters)
    def forward(self, x):
        x_skip = x
        x = self.layer_norm(x)
        x = self.conv(x)
        x = nd.relu(x)
        x = x_skip + x
        return x

class SelfAttetionBlock(Block):
    def __init__(self, **kwargs):
        super(SelfAttetionBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.layer_norm = nn.LayerNorm()

    def forward(self, x):
        x_skip = x
        x = self.layer_norm(x)
        q = x
        k = x
        v = x
        dk = nd.sqrt(len(nd.shape(k)))
        qk  = nd.softmax(q * k.T * 1. / dk)
        qkv = qk*v
        x = qkv + x_skip
        return x

class DenseBlock(Block):
    def __init__(self, num_units, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.layer_norm = nn.LayerNorm()
            self.dense = nn.Dense(num_units)

    def forward(self, x):
        x_skip = x
        x = self.layer_norm(x)
        x = self.dense(x)
        x = x + x_skip
        return x

class EncoderBlock(Block):
    def __init__(self, num_units, batch_size, sentence_size, embedding_size, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.sentence_size = sentence_size
        self.embedding_size = embedding_size
        with self.name_scope():
            self.inner_block = InnerEncoderBlock(num_units)
            self.self_attetion_block = SelfAttetionBlock()
            self.dense_block = DenseBlock(num_units)

    def position_encoding(self, x):
        """
        Position Encoding described in section 4.1 of
        End-To-End Memory Networks (https://arxiv.org/abs/1503.08895).
        """
        encoding = np.ones((self.sentence_size, self.embedding_size), dtype=np.float32)
        ls = sentence_size + 1
        le = embedding_size + 1
        for k in range(1, le):
            for j in range(1, ls):
                encoding[j-1, k-1] = (1.0 - j/float(ls)) - (
                    k / float(le)) * (1. - 2. * j/float(ls))
        encoding = nd.tile(encoding, reps=(self.batch_size, 1, 1))
        x = encoding + x
        return x

    def forward(self, x):
        x = position_encoding(x)
        x = self.inner_block(x)
        x = self.self_attetion_block(x)
        x = self.dense_block(x)
        return x

def ContextQueryAttention(Block):
    def __init__(self, num_units, **kwargs):
        super(ContextQueryAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(num_units)
    def forward(self, c, q):
        x = nd.concat(c, q, c*q)
        S = self.dense(x)
        S_bar = nd.softmax(S, axis=1)
        S_2bar = nd.softmax(S, axis=2)
        A = S_bar * Q.T
        B = S_bar * S_2_bar.T * c.T
        return A, B

class ModelEncoder(Block):
    def __init__(self, num_units, batch_size, sentence_size, embedding_size, **kwargs):
        super(ModelEncoder, self).__init__(**kwargs)
        with self.name_scope():
            self.context_query_attention = ContextQueryAttention(num_units)
            self.encoder0 = EncoderBlock(num_units, batch_size, sentence_size, embedding_size)
            self.encoder1 = EncoderBlock(num_units, batch_size, sentence_size, embedding_size)
            self.encoder2 = EncoderBlock(num_units, batch_size, sentence_size, embedding_size)

    def forward(self, C, Q):
        A, B = self.context_query_attention(C, Q)
        x = nd.concat(C, A, C * A, C * B)
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0)
        enc2 = self.encoder2(enc2)
        return enc0, enc1, enc2

class OutputLayer(Block):
    def __init__(self, num_units, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(num_units)
            self.weight = self.params.get(
                'weight', init=mx.init.Xavier(magnitude=2.24),
                shape=(num_units, num_units))

    def forward(self, enc1, enc2):
        x = nd.concat(enc1, enc2)
        x = self.dense(x)
        x = nd.log_softmax(x)
        return x

class QANet(Block):
    def __init__(self, num_units, batch_size, sentence_size, embedding_size, **kwargs):
        super(QANet, self).__init__(**kwargs)
        with self.name_scope():
            self.context_encoder = EncoderBlock(num_units, batch_size, sentence_size, embedding_size)
            self.query_encoder = EncoderBlock(num_units, batch_size, sentence_size, embedding_size)
            self.model_encoder = ModelEncoder(num_units, batch_size, sentence_size, embedding_size)
            self.output_start = OutputLayer(num_units)
            self.output_end = OutputLayer(num_units)

    def forward(self, c, q):
        c = self.context_encoder(c)
        q = self.query_encoder(q)
        enc0, enc1, enc2 = self.model_encoder(c, q)
        start = self.output_start(enc0, enc1)
        end = self.output_end(enc0, enc2)
        return start, end

class LogLoss(Loss):
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(LogLoss, self).__init__(weight, batch_axis, **kwargs)

    def forward(self, p1, p2):
        loss = -nd.mean(p1+p2, axis=self._batch_axis, exclude=True)
        return loss
