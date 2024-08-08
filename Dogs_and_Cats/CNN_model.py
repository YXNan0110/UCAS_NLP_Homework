import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader
from PIL import Image
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as data

file_path = './data'
model_path = './model/'
batch_size = 32
lr = 0.0002
epoch_num = 20
IMAGE_SIZE = 200

# 定义一个转换关系，用于将图像数据转换成PyTorch的Tensor形式
dataTransform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


class CatAndDogDataset(data.Dataset):
    def __init__(self, mode, dir):
        self.mode = mode
        self.list_img = []
        self.list_label = []
        self.data_size = 0
        self.transform = dataTransform

        if self.mode == 'train':
            dir = dir + '/train/'
        elif self.mode == 'test':
            dir = dir + '/val/'
        else:
            print('Undefined Dataset!')

        for file in os.listdir(dir):
            self.list_img.append(dir + file)    # 将图片路径和文件名添加至list_img
            self.data_size += 1
            name = file.split(sep='.')          # 分割文件名

            if name[0] == 'cat':
                self.list_label.append(0)       # 图片为猫，label为0
            else:
                self.list_label.append(1)       # 图片为狗，label为1

    def __getitem__(self, item):
        if self.mode == 'train':                                    # 训练集模式下需要读取数据集的image和label
            img = Image.open(self.list_img[item])
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor([label])
        elif self.mode == 'test':                                   # 测试集只需读取image
            img = Image.open(self.list_img[item])
            return self.transform(img)
        else:
            print('None')

    def __len__(self):
        return self.data_size


train_data = CatAndDogDataset('train', file_path)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_data = CatAndDogDataset('test', file_path)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 5, padding=2)
        self.conv3 = torch.nn.Conv2d(32, 32, 5, padding=2)
        self.linear1 = torch.nn.Linear(in_features=25*25*32, out_features=1024)
        self.linear2 = torch.nn.Linear(in_features=1024, out_features=128)
        self.linear3 = torch.nn.Linear(in_features=128, out_features=16)
        self.linear4 = torch.nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # batch_size*16*200*200
        x = F.max_pool2d(x, 2)  # batch_size*16*100*100
        x = F.relu(self.conv2(x))  # batch_size*32*100*100
        x = F.max_pool2d(x, 2)  # batch_size*32*50*50
        x = F.relu(self.conv3(x))  # batch_size*32*50*50
        x = F.max_pool2d(x, 2)  # batch_size*32*25*25
        x = x.view(x.size()[0], -1)  # batch_size*20000
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

model = CNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)    # 将模型迁移到gpu中

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_list = []
def train(epoch):
    running_loss = 0.0
    count = 0
    for inputs, target in train_loader:
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target.squeeze())
        loss.backward()
        optimizer.step()
        count += 1
        running_loss += loss.item()
        if count % 20 == 0:    # 每训练20个batch size输出一次均值loss
            print("[%d, %5d] loss: %.3f"%(epoch+1, count, running_loss/20))
            loss_list.append(running_loss/20)
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():    # 不需要计算梯度，只需要计算正确率
        for index in range(test_data.data_size):
            image = test_data.__getitem__(index)
            image = image.unsqueeze(0)
            target = test_data.list_label[index]
            image = image.to(device)    # 数据迁移
            outputs = model(image)
            _, predicted = torch.max(outputs.data, dim=1) # 求最大值的下标，沿第一个维度去找，即每一行的最大值，_代表的是该数值，返回的第二个参数是下标
            total += 1
            correct += (predicted == target).sum().item()
            if index % 50 == 0:
                print("Target:", target, "Output:", predicted)
    print("Accuracy: %d %% [%d/%d]" % (100 * correct / total, correct, total))

if __name__ == '__main__':
    for epoch in range(epoch_num):
        train(epoch)
    torch.save(model.state_dict(), os.path.join(model_path, 'cnn_model.pth'))
    plt.plot(loss_list, label='loss')
    plt.legend()
    plt.show()
    model.load_state_dict(torch.load('./model/cnn_model.pth'))
    test()
