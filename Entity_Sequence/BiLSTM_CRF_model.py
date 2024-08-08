import torch
from TorchCRF import CRF
from tqdm import tqdm
import os
from sklearn.metrics import precision_score, recall_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(1)
torch.cuda.init()

filepath = 'data'
epochs = 20
batch_size = 128
lr = 0.01
embedding_dim = 16
hidden_dim = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

pad_token = '<PAD>'
pad_id = 0
unk_token = '<UNK>'
unk_id = 1
tag2idx = {'<PAD>': 0, 'O': 1, 'B-LOC': 2, 'I-LOC': 3, 'B-PER': 4, 'I-PER': 5, 'B-ORG': 6, 'I-ORG': 7}
word_to_id = {'<PAD>': 0, '<UNK>': 1}
tags_num = 8


def dataset(data_dir):
    word = []
    tag = []
    with open(data_dir, encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line == '\n' and len(word_list) != 0:
                assert len(word_list) == len(tag_list)
                word.append(word_list)
                tag.append(tag_list)
                word_list = []
                tag_list = []
            else:
                line = line.strip().split(' ')
                word_list.append(line[0])
                tag_list.append(line[1])
    return word, tag


train_word, train_tag = dataset(filepath + '/train.txt')
test_word, test_tag = dataset(filepath + '/test.txt')


def build_vocab(sentences, word_to_id):
    for sentence in sentences:  # 建立word到索引的映射
        for word in sentence:
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)
    return word_to_id


word2idx = build_vocab(train_word, word_to_id)


def convert_to_ids_and_padding(seqs, to_ids):
    ids = []
    for seq in seqs:
        if len(seq) >= 70:    # 截断
            ids.append([to_ids[w] if w in to_ids else unk_id for w in seq[:70]])
        else:    # padding
            ids.append([to_ids[w] if w in to_ids else unk_id for w in seq] + [0]*(70-len(seq)))

    return torch.tensor(ids, dtype=torch.long)


def load_data(sentences, tags, word_to_idx, tag_to_id):

    sentences_pad = convert_to_ids_and_padding(sentences, word_to_idx)
    tags_pad = convert_to_ids_and_padding(tags, tag_to_id)

    dataset = torch.utils.data.TensorDataset(sentences_pad, tags_pad)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


train_dataloader = load_data(train_word, train_tag, word2idx, tag2idx)
test_dataloader = load_data(test_word, test_tag, word2idx, tag2idx)


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_id)
        self.Bi_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size // 2, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_size, tags_num)
        self.crf = CRF(num_tags=tags_num, batch_first=True)

    def forward(self, input):
        self.Bi_LSTM.flatten_parameters()
        embeds = self.embedding(input)
        lstm_out, _ = self.Bi_LSTM(embeds, None)
        logits = self.hidden2tag(lstm_out)
        return logits

    # 计算CRF条件对数似然，并返回其负值作为loss
    def crf_neg_log_likelihood(self, outputs, tags, mask):
        crf_llh = self.crf(outputs, tags, mask, reduction='mean')   # 根据BiLSTM模型输出、标签和mask计算条件对数似然
        return -crf_llh

    # 对输入序列进行 CRF 解码，返回最优的标签序列
    def crf_decode(self, emissions, mask):
        return self.crf.decode(emissions=emissions, mask=mask)


def train(epoch):
    loss_list = []
    Loss = 0
    count = 0
    for sentences, tags in tqdm(train_dataloader):
        sentences = sentences.to(device)  # sentences: batch_size*max_size, 经过了<PAD>
        tags = tags.to(device)  # size同sentences
        mask = torch.logical_not(torch.eq(sentences, torch.tensor(0)))    # mask表示是否是PAD
        model.train()
        # 第一步，pytorch梯度累积，需要清零梯度
        optimizer.zero_grad()
        # 第二步，得到loss
        outputs = model(sentences)
        loss = model.crf_neg_log_likelihood(outputs, tags, mask=mask)
        Loss += loss.item()
        if (count % 100) == 99:
            print("loss", Loss/count)
        loss.backward()
        optimizer.step()
        count += 1
    print("Epoch:", epoch + 1, "Loss:", Loss / count)
    loss_list.append(Loss / count)
    return loss_list


def f_precision(trues, preds):
    precision_sum = 0
    class_count = 0
    for i in range(2, 8):
        true_positives = sum((true == i and pred == i) for true, pred in zip(trues, preds))
        false_positives = sum((true != i and pred == i) for true, pred in zip(trues, preds))
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
        precision_sum += precision
        class_count += 1

    precision_res = precision_sum / class_count if class_count != 0 else 0
    return precision_res


def f_recall(trues, preds):
    recall_sum = 0
    class_count = 0
    for i in range(2, 8):
        true_positives = sum((true == i and pred == i) for true, pred in zip(trues, preds))
        false_negatives = sum((true == i and pred != i) for true, pred in zip(trues, preds))
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        recall_sum += recall
        class_count += 1

    recall_res = recall_sum / class_count if class_count != 0 else 0
    return recall_res


def loss_curve(loss_list):
    epoch = range(1, len(Loss) + 1)
    plt.figure()
    plt.plot(epoch, Loss, 'bo-', label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def acc_curve(list1, list2):
    epoch = range(1, len(list1) + 1)
    plt.figure()
    plt.plot(epoch, list1, 'bo-', label='Accuracy without O')
    plt.plot(epoch, list2, 'ro-', label='Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()


def f1_curve(f1_list):
    epoch = range(1, len(f1_list) + 1)
    plt.figure()
    plt.plot(epoch, f1_list, 'bo-', label='F1 score')
    plt.title('F1 Score Curve')
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    model = BiLSTM_CRF(len(word2idx), hidden_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    Loss = []
    Precision_no_o = []
    Precision = []
    F1 = []
    for epoch in range(epochs):
        Loss.append(train(epoch))
        with torch.no_grad():
            model.eval()
            # 用于计算f1_score
            all_pre = []
            all_tag = []
            for dev_sentences, dev_tags in tqdm(test_dataloader):
                dev_sentences = dev_sentences.to(device)
                dev_tags = dev_tags.to(device)
                # 预测的结果
                mask = torch.logical_not(torch.eq(dev_sentences, torch.tensor(0)))
                dev_outputs = model(dev_sentences)
                loss = model.crf_neg_log_likelihood(dev_outputs, dev_tags, mask=mask)
                dev_pre_tag = model.crf_decode(dev_outputs, mask=mask)
                for pre in dev_pre_tag:
                    all_pre.extend(pre)
                #
                tags_without_pad = torch.masked_select(dev_tags, mask).cpu().numpy()
                #
                all_tag.extend(tags_without_pad)
            # 去掉o-0计算precision、recall和f1_score
            precision_no_o = f_precision(all_tag, all_pre)
            recall_no_o = f_recall(all_tag, all_pre)
            f1_no_o = (2 * precision_no_o * recall_no_o) / (precision_no_o + recall_no_o)
            print("precision of entity:", precision_no_o, "recall of entity:", recall_no_o, "f1_score of entity:", f1_no_o)
            Precision_no_o.append(precision_no_o)
            # 所有的precision, recall, f1_score
            precision = precision_score(all_tag, all_pre, average='micro')    # micro更关注整体性能
            recall = recall_score(all_tag, all_pre, average='micro')
            f1 = 2 * (precision * recall) / (precision + recall)
            print("precision:", precision, "recall:", recall, "f1_score:", f1)
            Precision.append(precision)
            F1.append(f1_no_o)
    torch.save(model.state_dict(), os.path.join(filepath, 'model.pth'))
    Loss_List = []
    for i in Loss:
        for j in i:
            Loss_List.append(j)
    loss_curve(Loss_List)
    acc_curve(Precision_no_o, Precision)
    f1_curve(F1)


print("End")






