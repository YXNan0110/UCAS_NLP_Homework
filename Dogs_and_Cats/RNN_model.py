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
lr = 0.0001
epoch_num = 25
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


class RNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5, padding=2)  # 16*200*200 -> 16*100*100
        self.conv2 = torch.nn.Conv2d(16, 16, 5, padding=2)  # 16*100*100 -> 16*50*50
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], 1, -1)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        self.rnn.flatten_parameters()
        out, hn = self.rnn(x, h0.detach())
        out = self.linear(out[:, -1, :])
        return out

model = RNN(input_dim=40000, hidden_dim=100, layer_dim=2, output_dim=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

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
            total += 1
            correct += (predicted == target).sum().item()
            if index % 50 == 0:
                print("Target:", target, "Output:", predicted)
    print("Accuracy: %d %% [%d/%d]" % (100 * correct / total, correct, total))

if __name__ == '__main__':
    for epoch in range(epoch_num):
        train(epoch)
    torch.save(model.state_dict(), os.path.join(model_path, 'rnn_model.pth'))
    plt.plot(loss_list, label='loss')
    plt.legend()
    plt.show()
    model.load_state_dict(torch.load('./model/rnn_model.pth'))
    test()