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
lr = 0.001
epoch_num = 30
IMAGE_SIZE = 200

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
            self.list_img.append(dir + file)
            self.data_size += 1
            name = file.split(sep='.')

            if name[0] == 'cat':
                self.list_label.append(0)
            else:
                self.list_label.append(1)

    def __getitem__(self, item):
        if self.mode == 'train':
            img = Image.open(self.list_img[item])
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor([label])
        elif self.mode == 'test':
            img = Image.open(self.list_img[item])
            return self.transform(img)
        else:
            print('None')

    def __len__(self):
        return self.data_size


train_data = CatAndDogDataset('train', file_path)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_data = CatAndDogDataset('test', file_path)


class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.linear1 = torch.nn.Linear(3*200*200, 1024)
        self.linear2 = torch.nn.Linear(1024, 128)
        self.linear3 = torch.nn.Linear(128, 16)
        self.linear4 = torch.nn.Linear(16, 2)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

model = DNN()
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
        if count % 20 == 0:
            print("[%d, %5d] loss: %.3f"%(epoch+1, count, running_loss/20))
            loss_list.append(running_loss/20)
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for index in range(test_data.data_size):
            image = test_data.__getitem__(index)
            image = image.unsqueeze(0)
            target = test_data.list_label[index]
            image = image.to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs.data, dim=1)
            correct += (predicted == target).sum().item()
            if index % 50 == 0:
                print("Target:", target, "Output:", predicted)
    print("Accuracy: %d %% [%d/%d]" % (100 * correct / total, correct, total))

if __name__ == '__main__':
    for epoch in range(epoch_num):
        train(epoch)
    torch.save(model.state_dict(), os.path.join(model_path, 'dnn_model.pth'))
    plt.plot(loss_list, label='loss')
    plt.legend()
    plt.show()
    model.load_state_dict(torch.load('./model/dnn_model.pth'))
    test()