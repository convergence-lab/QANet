from model import QANet, LogLoss
from mxnet import gluon

def evaluate_exact_match_accuracy(data_iterator, net, ctx):
    for i (data, label) in enumrate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        pred_start, pred_end = net(data)

def train(num_units, batch_size, sentence_size, embedding_size, ctx):
    net = QANet(num_units, batch_size, sentence_size, embedding_size)
    net.collect_params().initialize(ctx)
    loss = LogLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
