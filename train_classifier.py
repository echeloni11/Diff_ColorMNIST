import numpy as np
import torch
import torch.utils.data as Data
from torchvision import datasets, transforms
import torch.nn.functional as F
import timeit

from network import Classifier
from dataset import add_hue_confounded, classifiedMNIST

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

rank = 0

# check availability of GPU and set the device accordingly

device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')


# define a transforms for preparing the dataset
transform = transforms.Compose([

        # transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        transforms.RandomRotation(10),      
        transforms.RandomAffine(5),
        
        # convert the image to a pytorch tensor
        transforms.ToTensor(), 
        
        # normalise the images with mean and std of the dataset
        # transforms.Normalize((0.1307,), (0.3081,)) 
        ])

tf = transforms.Compose([
        # convert the image to a pytorch tensor
        transforms.ToTensor(), 
        # normalise the images with mean and std of the dataset
        # transforms.Normalize((0.1307,), (0.3081,)) 
        ])

# Load the MNIST training, test datasets using `torchvision.datasets.MNIST` 
# using the transform defined above

train_dataset = datasets.MNIST('./data',train=True,transform=transform,download=True)
test_dataset =  datasets.MNIST('./data',train=False,transform=tf,download=True)

train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)

model = Classifier(input_channels=1).to(device)

losses_1 = []
losses_2 = []


def train(model, device, train_loader, optimizer, epoch, p_unif):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # send the image, target to the device
        data, target = data.to(device), target.to(device)
        # data, hues = add_hue_confounded(data, target, p_unif=p_unif)

        noise = torch.randn_like(data).to(device)
        noise_scale = torch.rand(1).to(device) * 0.2

        data = (data + noise * noise_scale).clip(0,1)
        
        # flush out the gradients stored in optimizer
        optimizer.zero_grad()
        # pass the image to the model and assign the output to variable named output
        output = model(data)
        # calculate the loss (use nll_loss in pytorch)
        loss = F.nll_loss(output, target)
        # do a backward pass
        loss.backward()
        # update the weights
        optimizer.step()
      
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            losses_1.append(loss.item())
            losses_2.append(100. * batch_idx / len(train_loader))

accuracy = []
avg_loss = []
def test(model, device, test_loader, p_unif):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
          
            # send the image, target to the device
            data, target = data.to(device), target.to(device)
            # data, hues = add_hue_confounded(data, target, p_unif=p_unif)
            # pass the image to the model and assign the output to variable named output
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
          
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    avg_loss.append(test_loss)
    accuracy.append(100. * correct / len(test_loader.dataset))


learning_rate = []
def adjust_learning_rate(optimizer, iter, each):
    # sets the learning rate to the initial LR decayed by 0.1 every 'each' iterations
    lr = 0.001 * (0.95 ** (iter // each))
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = lr
    optimizer.load_state_dict(state_dict)
    print("Learning rate = ",lr)
    return lr


## Define Adam Optimiser with a learning rate of 0.01
optimizer =  torch.optim.Adam(model.parameters(),lr=0.001)
p_unif = 0.05


start = timeit.default_timer()
for epoch in range(0, 61):
    lr = adjust_learning_rate(optimizer, epoch, 10)
    learning_rate.append(lr)
    train(model, device, train_dataloader, optimizer, epoch, p_unif)
    test(model, device, test_dataloader, p_unif=1)
    if epoch % 20 == 0:
        torch.save(model.state_dict(), f"./trained_classifiers/model_gray_noisy_{epoch}.pt")
stop = timeit.default_timer()
print('Total time taken: {} seconds'.format(int(stop - start)))