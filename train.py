import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
# from resnet import ResNet18
from readlabel import ISIC_2017
from resnet import *
import os


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    EPOCH = 10
    pre_epoch = 0
    BATCH_SIZE = 16
    LR = 0.01

    train_set = 'D:\dataset\ISIC2017\Train'
    train_label = 'D:\dataset\ISIC2017\Train_GT.csv'
    test_set = 'D:\dataset\ISIC2017\Test'
    test_label = 'D:\dataset\ISIC2017\Test_GT.csv'

    transform_train = transforms.Compose([transforms.Resize(512),
                                          transforms.CenterCrop(256),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([transforms.Resize(512),
                                         transforms.CenterCrop(256),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_dataset = ISIC_2017(data=train_set, label=train_label, transform=transform_train)
    test_dataset = ISIC_2017(data=test_set, label=test_label, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # for i, (input, target) in enumerate(trainloader):
    #     print('this is img',input[0])
    #     print('this is label',target)
    #     break
#
    classes = ('M', 'SK', 'NV')

    net = getresnet50(num_labels=3)
    best_acc = 0
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    #
    if os.path.isfile('D:\dataset\ISIC2017\model\state-best.tar'):
        print('=> loading checkpoint from D:\dataset\ISIC2017\model\state-best.tar')
        checkpoint = torch.load('D:\dataset\ISIC2017\model\state-best.tar')
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_acc = checkpoint['acc']

    print('Start Training, Resnet-50!')


    ## Test only
    net.eval()
    running_loss2 = 0.0
    running_corrects2 = 0
    label_0_total, label_0_count = 0,0
    label_1_total, label_1_count = 0,0
    label_2_total, label_2_count = 0,0


    for i, (input, target) in enumerate(testloader):
        input = input[0].cuda()
        target = target.cuda()

        optimizer.zero_grad()

        output = net(input)
        _, preds = torch.max(output, 1)
        # print(preds)

        # label 0
        # print(preds[torch.where(target == 0)])
        label_0_count += torch.sum(preds[torch.where(target == 0)] == 0)
        label_0_total += torch.sum(preds[torch.where(target == 0)] < 3)

        # label 1
        label_1_count += torch.sum(preds[torch.where(target == 1)] == 1)
        label_1_total += torch.sum(preds[torch.where(target == 1)] < 3)

        # label 2
        label_2_count += torch.sum(preds[torch.where(target == 2)] == 2)
        label_2_total += torch.sum(preds[torch.where(target == 2)] < 3)


        loss = criterion(output, target=target.resize(target.size(0)).long())

        running_loss2 += loss.item() * input.size(0)
        # running_corrects2 += torch.sum(preds == target.data)
        running_corrects2 += torch.sum(preds == target.resize(target.size(0)))

    # counting loss and acc in each epoch
    epoch_loss2 = running_loss2 / len(test_dataset)
    epoch_acc2 = running_corrects2.double() / len(test_dataset)
    print('in test process running_corrects is ', running_corrects2)
    print('in test process, length of dataset ', len(test_dataset))
    print('test****** Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss2, epoch_acc2))
    print('number ', label_0_count, label_0_total)
    print('number ', label_1_count, label_1_total)
    print('number ', label_2_count, label_2_total)
    print('test accuracy for label 0', (label_0_count / label_0_total).double())
    print('test accuracy for label 1', (label_1_count / label_1_total).double())
    print('test accuracy for label 2', (label_2_count / label_2_total).double())



    # for epoch in range(EPOCH):
    #     print('\nEpoch: %d' % (epoch + 1))
    #     net.train()
    #     running_loss = 0.0
    #     running_corrects = 0.0
    #
    #     for i, (data, target) in enumerate(trainloader):
    #         length = len(trainloader)
    #         inputs = data[0]
    #         labels = target
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #
    #         outputs = net(inputs)
    #         _, predicted = torch.max(outputs, 1)
    #         loss = criterion(outputs, target=labels.resize(labels.size(0)).long())
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss += loss.item() * inputs.size(0)
    #         running_corrects += torch.sum(predicted == labels.resize(labels.size(0)))
    #
    #     epoch_loss = running_loss / len(train_dataset)
    #     epoch_acc = running_corrects.double() / len(train_dataset)
    #     print('in train process running_corrects is ', running_corrects)
    #     print('in train process, length of dataset ', len(train_dataset))
    #     print('train****** Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    #
    #     # net.eval()
    #     running_loss2 = 0.0
    #     running_corrects2 = 0
    #
    #     for i, (input, target) in enumerate(testloader):
    #         input = input[0].cuda()
    #         target = target.cuda()
    #
    #         optimizer.zero_grad()
    #
    #         output = net(input)
    #         _, preds = torch.max(output, 1)
    #
    #         loss = criterion(output, target=target.resize(target.size(0)).long())
    #
    #         running_loss2 += loss.item() * input.size(0)
    #         # running_corrects2 += torch.sum(preds == target.data)
    #         running_corrects2 += torch.sum(preds == target.resize(target.size(0)))
    #
    #     # counting loss and acc in each epoch
    #     epoch_loss2 = running_loss2 / len(test_dataset)
    #     epoch_acc2 = running_corrects2.double() / len(test_dataset)
    #     print('in test process running_corrects is ', running_corrects2)
    #     print('in test process, length of dataset ', len(test_dataset))
    #     print('test****** Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss2, epoch_acc2))
    #
    #     if epoch_acc2 > best_acc:
    #         best_acc = epoch_acc2
    #         checkpoint_path = 'D:\dataset\ISIC2017\model\state-best.tar'
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': net.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': epoch_acc2,
    #             'acc': best_acc
    #         }, checkpoint_path)
    #
    #     print()

