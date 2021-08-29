import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from timeit import default_timer as timer

import numpy as np
import logging
from mgnet import MgNet


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def adjust_learning_rate(optimizer, epoch, init_lr):
    #lr = 1.0 / (epoch + 1)
    lr = init_lr * 0.1 ** (epoch // 50)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def load_data(path,minibatch_size,dataset):
    if dataset=='mnist':
        trainloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root=path, train=True, download=True,
                                         transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                          ])),
                                          batch_size=minibatch_size, num_workers=4,shuffle=True)
                                          
        testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root=path, train=False, download=True,
                                         transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                         ])),
                                         batch_size=minibatch_size, num_workers=4,shuffle=True)
        num_classes = 10
    if dataset=='cifar10':
        normalize = torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),\
                                        torchvision.transforms.RandomHorizontalFlip(),\
                                        torchvision.transforms.ToTensor(),\
                                        normalize])

        transform_test  = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),normalize])


        trainset = torchvision.datasets.CIFAR10(root=path, train=True,download=True,transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size,num_workers=4, shuffle=True)

        testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True,transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch_size,num_workers=4, shuffle=False)
        num_classes = 10
        
    if dataset=='cifar100':
        normalize = torchvision.transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),\
                                        torchvision.transforms.RandomHorizontalFlip(),\
                                        torchvision.transforms.ToTensor(),\
                                        normalize])

        transform_test  = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),normalize])


        trainset = torchvision.datasets.CIFAR100(root=path, train=True,download=True,transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size,num_workers=4, shuffle=True)

        testset = torchvision.datasets.CIFAR100(root=path, train=False, download=True,transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch_size,num_workers=4, shuffle=False)
        num_classes = 100
    
    return trainloader,testloader,num_classes

def save(model, optimizer,CHECKPOINT_NAME, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, CHECKPOINT_NAME)
    print(f'Save checkpoint from {CHECKPOINT_NAME}.')


def load(model, optimizer,CHECKPOINT_NAME):
    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
        return epoch
    return 0

def load_test(model,CHECKPOINT_NAME):
    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
        return epoch

def train_process(model,save_name,num_epochs,lr,trainloader,testloader):
    test_acc = []
    if use_cuda:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay = 0.0005)
    logger.info("total {} paramerters".format(sum(x.numel() for x in model.parameters())))

    start = timer()
    Test_acc = 0
    for epoch in range(1,num_epochs+1):
        current_lr = adjust_learning_rate(optimizer, epoch, lr)

        model.train()
        for i, (images, labels) in enumerate(trainloader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            # Forward pass to get the loss
            outputs = model(images) 
            loss = criterion(outputs, labels)

            # Backward and compute the gradient
            optimizer.zero_grad()
            loss.backward()  #backpropragation
            optimizer.step() #update the weights/parameters

      # Training accuracy 
        model.eval()
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(trainloader):
            with torch.no_grad():
                if use_cuda:
                    images = images.cuda()
                    labels = labels.cuda()  
                outputs = model(images)
                p_max, predicted = torch.max(outputs, 1) 
                total += labels.size(0)
                correct += (predicted == labels).sum()
        training_accuracy = float(correct)/total

        # Test accuracy
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(testloader):
            with torch.no_grad():
                if use_cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = model(images)
                p_max, predicted = torch.max(outputs, 1) 
                total += labels.size(0)
                correct += (predicted == labels).sum()
        test_accuracy = float(correct)/total
        test_acc.append(test_accuracy)
        if test_accuracy>Test_acc:
            save(model, optimizer,save_name,epoch)
            Test_acc = test_accuracy

        logger.info('Epoch:{}, lr:{:.6f}, train acc:{:.4f}, test acc:{:.4f}, best acc:{:.4f}'.\
                    format(epoch,current_lr,training_accuracy,test_accuracy,np.max(test_acc)))

    #end = timer()
    logger.info('last_train_acc:{:.4f},max_acc:{:.4f},last_acc:{:.4f}'.format(training_accuracy,np.max(test_acc),test_accuracy))
    return training_accuracy,np.max(test_acc),test_accuracy
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MgNet test')
    parser.add_argument('--dataset',type=str, default='cifar100')
    parser.add_argument('--path',type=str,default='./Data')
    parser.add_argument('--num-ite', type=str, help='The number of ite. in four level(layer). Use with 2,2,2,2 or 3,4,5,6.', default='2,2,2,2')
    parser.add_argument('--num-channel-u', type=int, help='number of channels of u', default=256)
    parser.add_argument('--num-channel-f', type=int, help='number of channels of f', default=256)
    parser.add_argument('--wise-B', action='store_true', help='different B in different grid')
    parser.add_argument('--minibatch-size',type=int,default=128)
    parser.add_argument('--num-epoch',type=int,default=300)
    parser.add_argument('--lr',type=float,default=0.1)
    parser.add_argument('--logger-name',type=str,default='./mgnet_test.log')
    
    args = parser.parse_args()
    
    args.num_ite = [int(i) for i in args.num_ite.split(',')]
    
    use_cuda = torch.cuda.is_available()
    print('Use GPU?', use_cuda)
    
    logger = get_logger(args.logger_name)
    save_name = './save_model/{}.pth'.format('test')
    logger.info(str(args))
    trainloader,testloader,num_classes = load_data(args.path,args.minibatch_size,args.dataset)
    model = MgNet(args,num_classes=num_classes)
    if use_cuda:
        model =model.cuda()
    train_acc,test_max_acc,test_last_acc = train_process(model,save_name,args.num_epoch,args.lr,trainloader,testloader)