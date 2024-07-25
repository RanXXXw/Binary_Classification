import torch
import torchvision
import torchvision.transforms as transforms
from net_runner import NetRunner


if __name__ == "__main__":

    train = True
    custom = False
    preview = True

    batch_size = 5

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(30),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    pt_trainset = torchvision.datasets.ImageFolder(root='.\dati\Train', transform=transform)
    pt_testset = torchvision.datasets.ImageFolder(root='.\dati\Test', transform=transform)

    trainset =  pt_trainset
    testset = pt_testset
    classes = pt_trainset.classes

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    runner = NetRunner(classes, batch_size)

    if train:
        runner.train(trainloader, preview)
    else:
        runner.test(testloader, True, preview)