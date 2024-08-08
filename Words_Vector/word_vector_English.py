import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange, tqdm
from sklearn.decomposition import PCA

lr = 0.001
epoch = 20
embedding_dim = 100


with open('data/en.txt') as f:
    text = f.read()
text = text.split()    # 分好词的英文list

# 将文本向量转换为对应数字向量
def word2index(context, word_to_idx):
    idx = [word_to_idx[w] for w in context]
    return torch.tensor(idx, dtype=torch.long)


word = set(text)    # 集合，去除重复的单词
set_len = len(word)
word_to_idx = {word: i for i, word in enumerate(word)}    # 词对应编号dict
idx_to_word = {i: word for i, word in enumerate(word)}    # 编号对应词dict

data = []

# 取上下文2个word作为样本，单词作为target，构建data训练集，eg：([text, text, text, text], target)
for i in range(2, len(text) - 2):
    context = [text[i - 2], text[i - 1],
               text[i + 1], text[i + 2]]
    target = text[i]
    data.append((context, target))

class CBOW(nn.Module):
    def __init__(self, set_len, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(set_len, embedding_dim)
        self.l1 = nn.Linear(embedding_dim, 128)
        self.l2 = nn.Linear(128, set_len)

    def forward(self, inputs):
        out = self.embeddings(inputs)    # (4, embedding_dim)
        out = sum(out)    # (1, embedding_dim)
        out = out.view(1, -1)
        out = F.relu(self.l1(out))
        out = self.l2(out)    # (1, dict_len)便于分类
        output = F.log_softmax(out, dim=-1)
        return output

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = CBOW(set_len, embedding_dim).to(device)
optimizer = optim.SGD(model.parameters(), lr)
loss_function = nn.NLLLoss()

loss_list = []

for i in range(epoch):
    total_loss = 0
    count = 0
    for context, target in tqdm(data):
        count += 1
        context_vector = word2index(context, word_to_idx).to(device)
        target = torch.tensor([word_to_idx[target]]).to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 开始前向传播
        train_predict = model(context_vector)
        loss = loss_function(train_predict, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        total_loss += loss.item()
#         if (count % 1000 == 0):
#             print("Epoch:", i+1, "Number:", count, "Loss:", total_loss/count)
    loss_list.append(total_loss/count)
    print("Epoch:", i+1, "Loss:", total_loss/count)

plt.plot(loss_list, label='loss')
plt.legend()
plt.show()

# 训练得到的词向量
word_vector = model.embeddings.weight.cpu().detach().numpy()

# 每个词：对应的embedding_dim词向量
word2vec = {}
for word in word_to_idx.keys():
    word2vec[word] = word_vector[word_to_idx[word], :]

# 使用pca进行降维得到(2*dict_len)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(word_vector)

# 降维后在生成一个词嵌入字典，即即{单词1:(维度一，维度二),单词2:(维度一，维度二)...}的格式，降维是为了画图
word2ReduceDimensionVec = {}
for word in word_to_idx.keys():
    word2ReduceDimensionVec[word] = principalComponents[word_to_idx[word], :]

# 将生成的字典写入到文件中，文件中实际记载的是完整的embedding_dim维词向量
with open("CBOW_en_wordvec.txt", 'w') as f:
    for key in word_to_idx.keys():
        f.write('\n')
        f.writelines('"' + str(key) + '":' + str(word2vec[key]))
    f.write('\n')

# 将词向量可视化
plt.figure(figsize=(20, 20))
# 只画出500个，太多显示效果很差
count = 0
for word, word_vec in word2ReduceDimensionVec.items():
    if count < 500:
        plt.scatter(word_vec[0], word_vec[1])
        plt.annotate(word, (word_vec[0], word_vec[1]))
        count += 1
plt.show()



