import os
import random
import time
import jieba
import nltk
import numpy as np
import torch
import re
from collections import Counter
import utils
from tqdm import tqdm
from model import Transformer, CustomSchedule, create_mask, Encoder, Decoder
from nltk.translate.bleu_score import corpus_bleu

data_path = 'en-zh'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
epochs = 1
save_dir = 'save'
save_filename = "model_state.pth"

PAD = 0
UNK = 1
num_layers = 6
d_model = 256
num_heads = 8
dff = 1024
beam_size = 5
max_beam_search_length = 100


# 设置随机种子
def seed_setting():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


seed_setting()
nltk.download('punkt')


# 去除语料中的标点符号
def remove_punctuation(text):
    punctuation_set = set(',.!?' + '，。！？')
    return re.sub(r'[' + re.escape(''.join(punctuation_set)) + ']+', ' ', text)


# 读取数据集
def data_reader(mode):
    src_file = os.path.join(data_path, f"{mode}.zh")
    trg_file = os.path.join(data_path, f"{mode}.en")
    src_sentences = []
    trg_sentences = []
    # 中文语料处理
    with (open(src_file, encoding='utf-8') as f):
        for line in f:
            line = remove_punctuation(line.strip())
            if mode == 'test':
                src_sentences.append(list(jieba.cut(line)))
            else:
                src_sentences.append(["BOS"] + list(jieba.cut(line)) + ["EOS"])
    # 英文语料处理
    with open(trg_file, encoding='utf-8') as f:
        for line in f:
            line = remove_punctuation(line.strip())
            trg_sentences.append(["BOS"] + nltk.word_tokenize(line.lower()) + ["EOS"])

    # 确保数量一致
    assert len(src_sentences) == len(trg_sentences), "Number of sentences does not match."

    return src_sentences, trg_sentences


train_zh, train_en = data_reader('train')
val_zh, val_en = data_reader('valid')
test_zh_word, test_en = data_reader('test')


def build_dict(sentences, max_words=5e4):
    """
    构造分词后的列表数据
    构建单词-索引映射（key为单词，value为id值）
    """
    # 统计数据集中单词词频
    word_count = Counter([word for sent in sentences for word in sent if word != ' '])
    # 按词频保留前max_words个单词构建词典
    # 添加UNK和PAD两个单词
    ls = word_count.most_common(int(max_words))
    total_words = len(ls) + 2
    word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
    word_dict['UNK'] = UNK
    word_dict['PAD'] = PAD
    # 构建id2word映射
    index_dict = {v: k for k, v in word_dict.items()}
    return word_dict, total_words, index_dict


word2idx_zh, dict_zh_len, idx2word_zh = build_dict(train_zh)
word2idx_en, dict_en_len, idx2word_en = build_dict(train_en)


def word2idx(en, zh, en_dict, cn_dict, sort=True):
    """
    将英文、中文单词列表转为单词索引列表
    `sort=True`表示以英文语句长度排序，以便按批次填充时，同批次语句填充尽量少
    """
    length = len(en)
    # 单词映射为索引
    out_en_ids = [[en_dict.get(word, UNK) for word in sent] for sent in en]
    out_zh_ids = [[cn_dict.get(word, UNK) for word in sent] for sent in zh]
    # 按相同顺序对中文、英文样本排序
    if sort:
        # 以英文语句长度排序
        sorted_index = utils.sort_len(out_en_ids)
        out_en_ids = [out_en_ids[idx] for idx in sorted_index]
        out_zh_ids = [out_zh_ids[idx] for idx in sorted_index]
    return out_en_ids, out_zh_ids


# 得到文字转换成数字的中英文语料
train_en, train_zh = word2idx(train_en, train_zh, word2idx_en, word2idx_zh)
val_en, val_zh = word2idx(val_en, val_zh, word2idx_en, word2idx_zh)
test_en, test_zh = word2idx(test_en, test_zh_word, word2idx_en, word2idx_zh)


def split_batch(en, zh, batch_size, shuffle=True):
    """
    划分批次
    `shuffle=True`表示对各批次顺序随机打乱
    """
    # 每隔batch_size取一个索引作为后续batch的起始索引
    idx_list = np.arange(0, len(en), batch_size)
    # 起始索引随机打乱
    if shuffle:
        np.random.shuffle(idx_list)
    # 存放所有batch的语句索引(batch_num * batch_size)
    batch_indexs = []
    for idx in idx_list:
        batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))
    # 构建批次列表
    batches = []
    for batch_index in batch_indexs:
        # 按当前批次的样本索引采样
        batch_en = [en[index] for index in batch_index]
        batch_zh = [zh[index] for index in batch_index]
        # 对当前批次中所有语句填充、对齐长度
        batch_zh = utils.seq_padding(batch_zh)    # batch_size * max_length_of_batch
        batch_en = utils.seq_padding(batch_en)    # batch_size * max_length_of_batch
        # 将当前批次添加到批次列表
        # Batch类用于实现注意力掩码
        batches.append(utils.Batch(batch_en, batch_zh))
    return batches


train_data = split_batch(train_en, train_zh, batch_size)
val_data = split_batch(val_en, val_zh, batch_size)

for sentence in test_zh:
    sentence.insert(0, word2idx_zh['BOS'])
    sentence.append(word2idx_zh['EOS'])

model = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, input_vocab_size=dict_en_len,
                    target_vocab_size=dict_zh_len, pe_input=6000, pe_target=10000)    # 英文到中文的翻译
model = model.to(device)
# print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
lr_scheduler = CustomSchedule(optimizer, d_model, warm_steps=4000)
criterion = torch.nn.CrossEntropyLoss(reduction='none')


def evaluate(model, criterion, val_data, epoch):
    model.eval()
    total_loss = 0
    batch_num = 0
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_data), total=len(val_data), desc=f"Epoch {epoch + 1} - Valuing"):
            # enc_padding_mask(batch_size * 1 * 1 * max_length_en_batch)
            # combined_mask(batch_size * 1 * max_length_zh_batch * max_length_zh_batch)
            # dec_padding_mask(batch_size * 1 * 1 * max_length_en_batch)
            enc_padding_mask, combined_mask, dec_padding_mask = create_mask(batch.src, batch.trg)
            enc_padding_mask = enc_padding_mask.to(device)
            combined_mask = combined_mask.to(device)
            dec_padding_mask = dec_padding_mask.to(device)

            # src-输入 trg-带<BOS>的目标 trg_y-不带<BOS>的目标
            out, _ = model(batch.src, batch.trg, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = utils.mask_loss_func(batch.trg_y, out, criterion)
            total_loss += loss.item()
            batch_num += 1
    print('Epoch:', epoch + 1, 'Valuing Loss:', total_loss / batch_num)


def train(model, optimizer, criterion, train_data):
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        model.train()
        total_loss = 0
        batch_num = 0
        for i, batch in tqdm(enumerate(train_data), total=len(train_data), desc=f"Epoch {epoch + 1} - Training"):
            time.sleep(0.1)
            # enc_padding_mask(batch_size * 1 * 1 * max_length_en_batch)
            # combined_mask(batch_size * 1 * max_length_zh_batch * max_length_zh_batch)
            # dec_padding_mask(batch_size * 1 * 1 * max_length_en_batch)
            enc_padding_mask, combined_mask, dec_padding_mask = create_mask(batch.src, batch.trg)
            enc_padding_mask = enc_padding_mask.to(device)
            combined_mask = combined_mask.to(device)
            dec_padding_mask = dec_padding_mask.to(device)

            optimizer.zero_grad()
            # src-输入 trg-带<BOS>的目标 trg_y-不带<BOS>的目标
            out, _ = model(batch.src, batch.trg, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = utils.mask_loss_func(batch.trg_y, out, criterion)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            lr_scheduler.step()
            batch_num += 1
        print('Epoch:', epoch+1, 'Training Loss:', total_loss / batch_num)
        evaluate(model, criterion, val_data, epoch)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(model.state_dict(), save_path)


train(model, optimizer, criterion, train_data)
save_path = os.path.join(save_dir, save_filename)
torch.save(model.state_dict(), save_path)


# model = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, input_vocab_size=dict_en_len, target_vocab_size=dict_zh_len, pe_input=6000, pe_target=10000)
model.load_state_dict(torch.load(save_path))
model.to(device)
model.eval()
top_hypotheses = []


def test():
    test_iteration = tqdm(zip(test_en, test_zh), desc='test bleu')
    with torch.no_grad():
        for idx, data in enumerate(test_iteration):
            time.sleep(0.1)
            en_sent = data[0]
            zh_sent = data[1]
            en = torch.from_numpy(np.array(en_sent).reshape(1, -1)).long().to(device)
            zh = torch.from_numpy(np.array(zh_sent).reshape(1, -1)).long().to(device)
            en_len = torch.from_numpy(np.array([len(en_sent)])).long().to(device)
            zh_len = torch.from_numpy(np.array([len(zh_sent)])).long().to(device)
            bos = torch.Tensor([[word2idx_zh['BOS']]]).long().to(device)
            completed_hypotheses = utils.beam_search(en, en_len, zh, zh_len,
                                                     bos, word2idx_zh['EOS'],
                                                     encoder=Encoder(num_layers, d_model, num_heads, dff, input_vocab_size=dict_en_len, maximun_position_encoding=6000, rate=0.1),
                                                     decoder=Decoder(num_layers, d_model, num_heads, dff, target_vocab_size=dict_zh_len, maximum_position_encoding=10000, rate=0.1),
                                                     topk=beam_size,
                                                     max_length=max_beam_search_length)
            top_hypotheses.append([idx2word_zh[id] for id in completed_hypotheses[0].value])

    bleu_score = corpus_bleu([[ref] for ref in test_zh_word], top_hypotheses)

    print('Corpus BLEU: {}'.format(bleu_score * 100))


test()




