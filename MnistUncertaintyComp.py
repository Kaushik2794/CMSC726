import torch
import torchvision
import matplotlib.pyplot as plt
import foolbox
import math
from PIL import Image

import scipy.stats as sps
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
        
n_epochs = 4
batch_size_train = 64
batch_size_test = 1
learning_rate = 0.01
momentum = 0.5
log_interval = 10
test_image_num = 0
target_num = 2
max_eps = 1

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
var_model.cuda()
                      
optimizer = optim.Adam(var_model.parameters(), lr=0.01)

def pred_entropy(vec):
    entropy = 0
    for x in np.nditer(vec):
        if x != 0:
            entropy -= math.exp(x) * x / math.log(2)
    return entropy

def train(epoch):
    var_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
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

for epoch in range(1, n_epochs):
    train(epoch)
      
      
#examples = enumerate(test_loader)
#batch_idx, (data, target) = next(examples)

mnist_trainset = torchvision.datasets.MNIST(root='./files', train=False, download=True, transform=None)
test_image_zero, test_target_zero = mnist_trainset[test_image_num]
while(test_target_zero != target_num):
    test_image_num += 1
    test_image_zero, test_target_zero = mnist_trainset[test_image_num]
print("Test index chosen: ",test_image_num)

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
    data, target = data.cuda(), target.cuda()
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
adversarial_target = attack(data, target, max_epsilon=max_eps)

trials = 100
var_ratios = np.zeros(11)
entropies = np.zeros(11)
mi = np.zeros(11)
for j in range (0,11):
    adversarial = data[0] + (adversarial_target[0] - data[0])*(j/10.0)
    im = Image.fromarray(np.uint8(adversarial*255))
    adversarial = loader(im).float()
    adversarial = Variable(adversarial, requires_grad=True)
    adversarial = adversarial.unsqueeze(0)  
    results = np.zeros(trials, dtype = int)
    vec = np.zeros((1,10))
    ent = 0
    for i in range (trials):
        attack_data = adversarial.cuda()
        var_model.draw()
        result = var_model(attack_data).detach()
        # calculate entropy measures
        temp_vec = result.cpu().numpy()
        vec += np.exp(temp_vec)
        ent += pred_entropy(temp_vec)
        
        values, index = torch.max(result,1)
        results[i] = index
        conf = np.exp(result[0][index])
        #print(index[0], conf)

    vec /= trials
    entropies[j] = pred_entropy(np.log(vec))
    mi[j] = entropies[j] - ent/trials    
    
    moderesult = sps.mode(results)
    print(sps.mode(results))
    var_ratios[j] = 1 - moderesult[1][0]/trials
    if j == 10:
        plt.figure(1)
        plt.imshow(im)

print(var_ratios)


# Simple CNN with Dropout architecture
class DropNet(nn.Module):
    def __init__(self):
        super(DropNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Data loading code
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=batch_size_train, shuffle=True)


model = DropNet()
model = model.cuda()
            
optimizer = optim.Adam(model.parameters(), lr=0.01)
def trainDrop(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

    
for epoch in range(1, n_epochs):
    trainDrop(epoch)
    
#examples = enumerate(test_loader)
#batch_idx, (data, target) = next(examples)

test_image_zero, test_target_zero = mnist_trainset[test_image_num]

target = test_target_zero

loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

data = loader(test_image_zero).float()
data = Variable(data, requires_grad=True)
data = data.unsqueeze(0)  


# Test prediction for dropnet
model.train()
print(target)
for i in range (1, 10):
    data, target = data.cuda(), target.cuda()
    result = model(data).detach()
    values, index = torch.max(result,1)
    conf = np.exp(result[0][index])
    print(index[0], conf)
        

# Convert data and target to Foolbox friendly format
data = data[0].data.cpu().numpy()
target = target.data.cpu().numpy()

var_model = model
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
adversarial_target = attack(data, target, max_epsilon=max_eps)

trials = 100
var_ratios2 = np.zeros(11)
var_model.train()

entropies2 = np.zeros(11)
mi2 = np.zeros(11)
for j in range (0,11):
    adversarial = data[0] + (adversarial_target[0] - data[0])*(j/10.0)
    im = Image.fromarray(np.uint8(adversarial*255))
    adversarial = loader(im).float()
    adversarial = Variable(adversarial, requires_grad=True)
    adversarial = adversarial.unsqueeze(0)  
    results = np.zeros(trials, dtype = int)
    vec = np.zeros((1,10))
    ent = 0
    for i in range (trials):
        attack_data = adversarial.cuda()
        result = var_model(attack_data).detach()
         # calculate entropy measures
        temp_vec = result.cpu().numpy()
        vec += np.exp(temp_vec)
        ent += pred_entropy(temp_vec)
        
        values, index = torch.max(result,1)
        results[i] = index
        conf = np.exp(result[0][index])
        #print(index[0], conf)
        
    vec /= trials
    entropies2[j] = pred_entropy(np.log(vec))
    mi2[j] = entropies2[j] - ent/trials

    moderesult = sps.mode(results)
    print(sps.mode(results))
    var_ratios2[j] = 1 - moderesult[1][0]/trials
    if j == 10:
        plt.figure(2)
        plt.imshow(im)

print(var_ratios2)

plt.figure(3)
plt.plot(np.arange(0,11)*0.1*max_eps, var_ratios, label='Bayes By Backprop')
plt.plot(np.arange(0,11)*0.1*max_eps, var_ratios2, label='Dropout')
plt.legend(loc='upper left')
plt.ylabel('Variation Ratio')
plt.xlabel('Epsilon in FGSM')
#plt.title('Uncertainty on MNIST')

plt.figure(4)
plt.plot(np.arange(0,11)*0.1*max_eps, entropies, label='Bayes By Backprop')
plt.plot(np.arange(0,11)*0.1*max_eps, entropies2, label='Dropout')
plt.legend(loc='upper left')
plt.ylabel('Predicted Entropy')
plt.xlabel('Epsilon in FGSM')
plt.title('Uncertainty on MNIST')


plt.figure(5)
plt.plot(np.arange(0,11)*0.1*max_eps, mi, label='Bayes By Backprop')
plt.plot(np.arange(0,11)*0.1*max_eps, mi2, label='Dropout')
plt.legend(loc='upper left')
plt.ylabel('Mutual Information')
plt.xlabel('Epsilon in FGSM')
#plt.title('Uncertainty on MNIST')

plt.show()



