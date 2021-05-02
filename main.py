import torch
import torch.nn as nn
import torch.optim as optim
import data
import models
import os
import time
import pandas as pd

# from util import *
try:
    from progress.bar import Bar as Bar
except:
    os.system('pip install progress')
    from progress.bar import Bar as Bar

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
# parser.add_argument('-t','--TASK', metavar='Task', default='A' ,type = str,
#                     choices=['A','B','C','D','E'] ,help='Task Name')
parser.add_argument('-e','--epochs', default=200, type=int, 
                    metavar='num_epochs',help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=36, type=int,
                    metavar='batch_size', help='mini-batch size (default: 36)') #default = 256

parser.add_argument('-lr', '--learning-rate', default=0.002, type=float,
                    metavar='lr', help='initial learning rate')

parser.add_argument('-lrs', '--learning-rate-schedule', default=2, type=float,
                    metavar='learning-rate-schedule', help='learning rate schedule')

parser.add_argument('-n','--num_classes', default=196, type=int,
                    metavar='num_classes', help='number of classes')

parser.add_argument('-i','--iterations', default=200, type=int,
                    metavar='input_size', help='number of classes')

parser.add_argument('-input','--input_size', default=224, type=int,
                    metavar='input_size', help='number of classes')

parser.add_argument('-dir','--data_dir', default="../hw2_dataset/", type=str,
                    metavar='data_dir', help='dir of the dataset')

parser.add_argument('-m','--model', default="", type=str,
                    metavar='model', help='model')

parser.add_argument('-d', '--data-augmentation', default=0, type=int,
                    metavar='W', help='data-augmentation method')

parser.add_argument('--out', default='', type=str,
                    metavar='W', help='Save directory')
#weight decay
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
#momentum
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


loss_train,acc_train,loss_test,acc_test = [],[],[],[]
#将数据写入文件
def writer():
    df = pd.DataFrame(columns = ['loss_train','acc_train','loss_test','acc_test'])
    df['loss_train'] = loss_train
    df['acc_train'] = acc_train
    df['loss_test'] = loss_test
    df['acc_test'] = acc_test
    df.to_csv("Model.csv")

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train_model(model,train_loader, valid_loader, criterion, optimizer ,scheduler = None,batch_size = 36, start_epoch = 0,num_epochs=200 ):
    def train(model, data_loader,optimizer,criterion, valid = False):
        if not valid:
            bar = Bar('Training', max=(1+(len(data_loader.dataset)-1)//batch_size))
            model.train(True)
        else:
            bar = Bar('Valid', max=(1+(len(data_loader.dataset)-1)//batch_size))
            model.train(False)
        
        total_loss = 0.0
        total_correct = 0

        for batch_idx,(inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)              #拷贝输入
            labels = labels.to(device)              #拷贝标签

            if not valid:
                optimizer.zero_grad()                   #0初始化梯度
            
            outputs = model(inputs)                 #进行一次迭代
            loss = criterion(outputs, labels)       #计算损失函数
            
            _, predictions = torch.max(outputs, 1)  #预测标签
            
            if not valid:
                loss.backward()                         #反向传播梯度
                optimizer.step()                        #优化器迭代

            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)

            # plot progress
            bar.suffix  = '({batch}/{size}) | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc:.4f}'.format(
                        batch=batch_idx + 1,
                        size=(1+(len(data_loader.dataset)-1)//batch_size),
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=loss.item(),
                        acc=torch.sum(predictions == labels.data)/inputs.size(0)
                        )
            bar.next()
        bar.finish()

        epoch_loss = total_loss / len(data_loader.dataset)
        epoch_acc = total_correct.double() / len(data_loader.dataset)

        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    for epoch in range(start_epoch,num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss, train_acc = train(model, train_loader,optimizer,criterion)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        # val_loss, val_acc = valid(model, valid_loader,criterion)
        val_loss, val_acc = train(model, valid_loader,optimizer,criterion ,valid=True)
        print("validation: {:.4f}, {:.4f}".format(val_loss, val_acc))
        #学习率递降方式
        if scheduler is not None:
            scheduler.step()
            print("Learning Rate is {}".format(scheduler.get_last_lr()))
        
        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict()
            }, is_best)
        
        # loss_train,acc_train,loss_test,acc_test = [],[],[],[]
        # writer metrics
        loss_train.append(train_loss)
        acc_train.append(train_acc)
        loss_test.append(val_loss)
        acc_test.append(val_acc)
        writer()


def save_checkpoint(state, is_best, checkpoint='result', filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)
    
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        best_model = model
        torch.save(best_model, os.path.join(checkpoint,'best_model.pt')) #模型保存

def choose_schedule(num,optimizer):
    '''
        num: method number
        optimizer : an object of torch.optim 
    '''
    if num == 0:
        scheduler = None
    elif num == 1:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 200, 260, 320 , 360], gamma=0.6)
    elif num == 2:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.4) #学习率递降策略
    elif num == 3:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 60, gamma=0.6) #学习率递降策略
    elif num == 4:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.1) #学习率递降策略
    elif num == 5:
        scheduler = optim.lr_sheduler.CosineAnnealingLR(optimizer, T_max = num_epochs, eta_min=0)
    
    return scheduler

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    global args
    args = parser.parse_args()
    print(args)
    print(model_names)

    ## about model
    num_classes = args.num_classes
    inupt_size = args.input_size
    batch_size = args.batch_size
    num_epochs = args.epochs
    start_epoch = 0
    iterations = args.iterations
    lr = args.learning_rate


    ## about data
    data_dir = args.data_dir
    data_augmentation = args.data_augmentation

    ## model initialization
    if args.model:
        try:
            model = getattr(models,args.model)(num_classes = num_classes)
        except:
            assert 0,"model is not exsit in models.py"
    else:
        model = models.ResNet50(num_classes=num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = data.load_data(batch_size=batch_size,input_size=inupt_size,data_augmentation = args.data_augmentation)

    ## optimizer
    # optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay= args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(),momentum= args.momentum, lr = args.learning_rate, weight_decay=args.weight_decay)

    #load_resume
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if os.path.isfile('Model.csv'):
            old_df = pd.read_csv('Model.csv')
            loss_train,acc_train,loss_test,acc_test = list(old_df['loss_train']),list(old_df['acc_train']),list(old_df['loss_test']),list(old_df['acc_test'])

    #set learning rate to lr
    adjust_learning_rate(optimizer,lr)

    # scheduler
    scheduler = choose_schedule(args.learning_rate_schedule,optimizer)


    import torchsummary
    print(torchsummary.summary(model, (3, inupt_size, inupt_size)))
    
    #计时
    import time
    time_start=time.time()

    ## loss function
    criterion = nn.CrossEntropyLoss().to(device)
    train_model(model,train_loader, valid_loader, criterion, optimizer, scheduler, batch_size = batch_size,start_epoch = start_epoch,num_epochs=num_epochs)

    time_end=time.time()
    print('train time cost',time_end-time_start,'s')




