import torch
import random
import numpy as np
import utils
import os
from utils import SemEvalDataLoader
from model import Att_BLSTM
import torch.nn as nn
from tqdm import tqdm
import time

data_dir = './SemEval2010_task8_all_data'
output_dir = './output'
embedding_path = './embedding/glove.6B.100d.txt'
word_dim = 100
seed = 5782
cuda = 0
epochs = 30
batch_size = 10
lr = 1  # learning rate
max_len = 100  # max length of sentence
emb_dropout = 0.3  # the possibility of dropout in embedding layer
lstm_dropout = 0.3  # the possibility of dropout in (Bi)LSTM layer
linear_dropout = 0.5  # the possibility of dropout in linear layer
hidden_size = 100  # the dimension of hidden units in (Bi)LSTM layer
layers_num = 1  # num of RNN layers
L2_decay = 1e-5  # L2 weight decay
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_setting(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


seed_setting(seed)

path_train = './SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
path_test = './SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'

utils.convert(path_train, 'train.json')
utils.convert(path_test, 'test.json')


def load_embedding():
    word2id = dict()  # word to wordID
    word_vec = list()  # wordID to word embedding

    word2id['PAD'] = len(word2id)  # PAD character
    word2id['UNK'] = len(word2id)  # out of vocabulary
    word2id['<e1>'] = len(word2id)
    word2id['<e2>'] = len(word2id)
    word2id['</e1>'] = len(word2id)
    word2id['</e2>'] = len(word2id)

    with open(embedding_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip().split()
            if len(line) != word_dim + 1:
                continue
            word2id[line[0]] = len(word2id)
            word_vec.append(np.asarray(line[1:], dtype=np.float32))

    word_vec = np.stack(word_vec)
    vec_mean, vec_std = word_vec.mean(), word_vec.std()
    special_emb = np.random.normal(vec_mean, vec_std, (6, word_dim))
    special_emb[0] = 0  # <pad> is initialize as zero

    word_vec = np.concatenate((special_emb, word_vec), axis=0)
    word_vec = word_vec.astype(np.float32).reshape(-1, word_dim)
    word_vec = torch.from_numpy(word_vec)
    return word2id, word_vec


# word2id-词典 word_vec 转换为embedding
word2id, word_vec = load_embedding()


def load_relation():
    relation_file = os.path.join(data_dir, 'relation2id.txt')
    rel2id = {}
    id2rel = {}
    with open(relation_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            relation, id_s = line.strip().split()
            id_d = int(id_s)
            rel2id[relation] = id_d
            id2rel[id_d] = relation
    return rel2id, id2rel, len(rel2id)


# 加载关系词典
rel2id, id2rel, class_num = load_relation()
# 数据加载器
loader = SemEvalDataLoader(rel2id, word2id)
train_loader = loader.get_train()
dev_loader = loader.get_dev()

model = Att_BLSTM(word_vec=word_vec, class_num=class_num, max_len=max_len, word_dim=word_dim, hidden_size=hidden_size,
                  layers_num=layers_num, emb_dropout=emb_dropout, lstm_dropout=lstm_dropout, linear_dropout=linear_dropout)
model = model.to(device)
criterion = nn.CrossEntropyLoss()


class Eval(object):
    def __init__(self):
        self.device = device

    def evaluate(self, model, criterion, data_loader):
        predict_label = []
        true_label = []
        total_loss = 0.0
        with torch.no_grad():
            model.eval()
            for _, (data, label) in enumerate(data_loader):
                data = data.to(self.device)
                label = label.to(self.device)

                logits = model(data)
                loss = criterion(logits, label)
                total_loss += loss.item() * logits.shape[0]

                _, pred = torch.max(logits, dim=1)  # replace softmax with max function, same impacts
                pred = pred.cpu().detach().numpy().reshape((-1, 1))
                label = label.cpu().detach().numpy().reshape((-1, 1))
                predict_label.append(pred)
                true_label.append(label)
        predict_label = np.concatenate(predict_label, axis=0).reshape(-1).astype(np.int64)
        true_label = np.concatenate(true_label, axis=0).reshape(-1).astype(np.int64)
        eval_loss = total_loss / predict_label.shape[0]

        f1 = utils.semeval_scorer(predict_label, true_label)
        return f1, eval_loss, predict_label


def train(model, criterion, train_loader, dev_loader):
    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=lr, weight_decay=L2_decay)

    print(model)
    print('traning model parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('%s :  %s' % (name, str(param.data.shape)))
    print('--------------------------------------')
    print('start to train the model ...')

    eval_tool = Eval()
    max_f1 = -float('inf')
    for epoch in range(1, epochs + 1):
        for step, (data, label) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch} - Training"):
            time.sleep(0.1)
            model.train()
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=5)
            optimizer.step()

        _, train_loss, _ = eval_tool.evaluate(model, criterion, train_loader)
        f1, dev_loss, _ = eval_tool.evaluate(model, criterion, dev_loader)

        print('[%03d] train_loss: %.3f | dev_loss: %.3f | micro f1 on dev: %.4f'
              % (epoch, train_loss, dev_loss, f1), end=' ')
        if f1 > max_f1:
            max_f1 = f1
            torch.save(model.state_dict(), os.path.join(output_dir, 'model.pkl'))
            print('>>> save models!')
        else:
            print()


train(model, criterion, train_loader, dev_loader)
test_loader = loader.get_test()


def test(model, criterion, test_loader):
    print('--------------------------------------')
    print('start test ...')

    model.load_state_dict(torch.load(
        os.path.join(output_dir, 'model.pkl')))
    eval_tool = Eval()
    f1, test_loss, predict_label = eval_tool.evaluate(model, criterion, test_loader)
    print('test_loss: %.3f | micro f1 on test:  %.4f' % (test_loss, f1))
    return predict_label


predict_label = test(model, criterion, test_loader)
utils.print_result(predict_label, id2rel)






