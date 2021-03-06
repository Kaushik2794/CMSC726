import torch
import torchvision
import matplotlib.pyplot as plt
import foolbox
from PIL import Image

import numpy as np
import pyvarinf
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# Simple CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)

    def forward(self, x):
        x = self.bn1(F.relu(F.max_pool2d(self.conv1(x), 2)))
        x = self.bn2(F.relu(F.max_pool2d(self.conv2(x), 2)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
        
# Simple CNN architecture_2
#class Net(nn.Module):
#    def _init_(self):
#        super(Net, self)._init_()
#        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#        self.convm1 = nn.Conv2d(10,10, kernel_size=3, padding=1)
#        self.convm2 = nn.Conv2d(10,10, kernel_size=3, padding=1)
#        self.convm3 = nn.Conv2d(10,10, kernel_size=3, padding=1)
#        self.fc1 = nn.Linear(500, 50)
#        self.fc2 = nn.Linear(50, 10)
#        self.bn1 = nn.BatchNorm2d(10)
#        self.bn2 = nn.BatchNorm2d(20)
#
#    def forward(self, x):
#        x = self.bn1(F.relu(F.max_pool2d(self.conv1(x), 2)))
#        x = self.bn1(F.relu(self.convm1(x)))
#        x = self.bn1(F.relu(self.convm2(x)))
#        x = self.bn1(F.relu(self.convm3(x)))
#        x = self.bn2(F.relu(F.max_pool2d(self.conv2(x), 2)))
#        x = x.view(-1, 500)
#        x = F.relu(self.fc1(x))
#        x = self.fc2(x)
#        return F.log_softmax(x)
    
    
n_epochs = 3
batch_size_train = 64
batch_size_test = 1
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Data loading code
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=batch_size_train, shuffle=True)


model = Net()
var_model = pyvarinf.Variationalize(model)

print(var_model)
#var_model.cuda()
                      
optimizer = optim.Adam(var_model.parameters(), lr=0.01)

def train(epoch):
    var_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = var_model(data)
        loss_error = F.nll_loss(output, target)
        
        loss_prior = var_model.prior_loss() / 60000
        loss = loss_error + loss_prior
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

for epoch in range(1, 4):
    train(epoch)
      
      
#examples = enumerate(test_loader)
#batch_idx, (data, target) = next(examples)

mnist_trainset = torchvision.datasets.MNIST(root='./files', train=False, download=True, transform=None)
test_image_zero, test_target_zero = mnist_trainset[1]

plt.figure(0)
plt.imshow(test_image_zero)

target = test_target_zero

loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

data = loader(test_image_zero).float()
data = Variable(data, requires_grad=True)
data = data.unsqueeze(0)  

# Test prediction
var_model.eval()
# Use PyVarInf's Sample wrapper to sample models from the variational distribution using draw()
var_model = pyvarinf.Sample(var_model)
print(target)
for i in range (1, 10):
    #data, target = data.cuda(), target.cuda()
    var_model.draw()
    result = var_model(data).detach()
    values, index = torch.max(result,1)
    conf = np.exp(result[0][index])
    print(index[0], conf)
        

# Convert data and target to Foolbox friendly format
data = data[0].data.cpu().numpy()
target = target.data.cpu().numpy()

var_model.eval()
# Convert model to Foolbox model
fmodel = foolbox.models.PyTorchModel(
    var_model, bounds=(0, 1), num_classes=10)

print('label', target)
result = np.exp(fmodel.predictions(data))
pred = np.argmax(result)

# Black Box Attack Model
print('predicted class', pred, '| confidence', result[pred]);
# apply attack on source image
attack = foolbox.attacks.FGSM(fmodel)
adversarial = attack(data, target)

im = Image.fromarray(np.uint8(adversarial[0]*255))
adversarial = loader(im).float()
adversarial = Variable(adversarial, requires_grad=True)
adversarial = adversarial.unsqueeze(0)  
for i in range (1, 10):
    attack_data = adversarial
    var_model.draw()
    result = var_model(attack_data).detach()
    values, index = torch.max(result,1)
    conf = np.exp(result[0][index])
    print(index[0], conf)

plt.figure(1)
plt.imshow(im)
plt.show()

