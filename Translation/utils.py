import torch
import numpy as np
from torch.autograd import Variable
from collections import namedtuple
from model import create_mask

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD = 0


def subsequent_mask(size):
    "Mask out subsequent positions."
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


# 按照语句长度排序
def sort_len(seq):
    # 返回原始序列中句子长度从短到长的序号列表
    return sorted(range(len(seq)), key=lambda x: len(seq[x]))


def seq_padding(X, padding=PAD):
    """
    按批次（batch）对数据填充、长度对齐
    """
    # 计算该批次各条样本语句长度
    L = [len(x) for x in X]
    # 获取该批次样本中语句长度最大值
    max_length_ = max(L)
    # 遍历该批次样本，如果语句长度小于最大长度，则用padding填充
    return np.array([
        np.concatenate([x, [padding] * (max_length_ - len(x))]) if len(x) < max_length_ else x for x in X
    ])


class Batch:
    """
    批次类
        1. 输入序列（源）
        2. 输出序列（目标）
        3. 构造掩码
    """

    def __init__(self, src, trg=None, pad=PAD):
        # 将输入、输出单词id表示的数据规范成整数类型
        src = torch.from_numpy(src).to(device).long()
        trg = torch.from_numpy(trg).to(device).long()
        self.src = src
        # 对于当前输入的语句非空部分进行判断，bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对解码器使用的目标语句进行掩码
        if trg is not None:
            # 解码器使用的目标输入部分
            self.trg = trg[:, : -1]    # 去除最后一列
            # 解码器训练时应预测输出的目标结果
            self.trg_y = trg[:, 1:]    # 去除第一列的<BOS>
            # 将目标输入部分进行注意力掩码
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的目标结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    # 掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)    # batch_size * 1 * max_length_of_batch
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))    # batch_size * max_length_of_batch * max_length_of_batch
        return tgt_mask


def mask_loss_func(real, pred, loss_function):
    # print(real.shape, pred.shape)
    # _loss = loss_object(pred, real) # [b, targ_seq_len]
    _loss = loss_function(pred.transpose(-1, -2), real)  # [b, targ_seq_len]

    # logical_not  取非
    # mask 每个元素为bool值，如果real中有pad，则mask相应位置就为False
    mask = torch.logical_not(real.eq(0)).type(_loss.dtype)

    # 对应位置相乘，token上的损失被保留了下来，pad的loss被置为0或False 去掉，不计算在内
    _loss *= mask
    return _loss.sum() / mask.sum().item()


def beam_search(en,en_lengths,zh, zh_length, y, EOS_id, encoder, decoder, topk=5,max_length=100):
    enc_padding_mask, combined_mask, dec_padding_mask = create_mask(en, zh)
    encoder.cuda()
    decoder.cuda()
    encoder_out = encoder(en, enc_padding_mask)

    BOS_id = y[0][0].item()
    hypotheses = [[BOS_id]]
    hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=y.device)
    completed_hypotheses = []
    t = 0
    while len(completed_hypotheses) < topk and t < max_length:
        t += 1
        hyp_num = len(hypotheses)
        # 扩展成batch
        exp_src_encodings = encoder_out.expand(hyp_num, encoder_out.shape[1], encoder_out.shape[2])
        exp_x_lengths = en_lengths.expand(hyp_num)
        dec_output, attention_weights = decoder(zh, encoder_out, combined_mask, dec_padding_mask)
        live_hyp_num = topk - len(completed_hypotheses)

        # 这里把num * vocab 展开来方便取topk
        contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand(hyp_num, dec_output.shape[-1]) + dec_output[:, -1, :].squeeze(1)).view(-1)
        top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

        # 标记当前概率最大的k个，其是跟在哪个单词的后面
        prev_hyp_ids = top_cand_hyp_pos / (dec_output.shape[-1])
        hyp_word_ids = top_cand_hyp_pos % (dec_output.shape[-1])

        new_hypotheses = []
        live_hyp_ids = []
        new_hyp_scores = []

        for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
            prev_hyp_id = int(prev_hyp_id.item())
            hyp_word_id = int(hyp_word_id.item())
            cand_new_hyp_score = cand_new_hyp_score.item()

            # 将当前最大概率的k个，拼接在正确的prev单词后面
            new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word_id]
            if hyp_word_id == EOS_id:
                # 搜寻终止
                completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                       score=cand_new_hyp_score))
            else:
                new_hypotheses.append(new_hyp_sent)
                live_hyp_ids.append(prev_hyp_id)
                new_hyp_scores.append(cand_new_hyp_score)

        if len(completed_hypotheses) == topk:
            break

        hypotheses = new_hypotheses
        hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=y.device)

    # 若搜寻了max_len后还没有一个到达EOS则取第一个
    if len(completed_hypotheses) == 0:
        completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                               score=hyp_scores[0].item()))
    completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

    return completed_hypotheses





