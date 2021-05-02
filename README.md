# Car196-ResNet152
Acc 88%

Prepare for the data:
```bash
mkdir data
cd data
wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz
wget http://imagenet.stanford.edu/internal/car196/cars_annos.mat
```

And run:
```bash
!python main.py -e 400 -b 36 -lr 0.0001 -m ResNet152 -d 1 --resume 'result/checkpoint.pth.tar'
```

You can also choose ResNet50 use :
```bash
!python main.py -e 400 -b 36 -lr 0.0001 -m ResNet50 -d 1 --resume 'result/checkpoint.pth.tar'
```

There tells the hard tuning process :
[Hard param tuning and a little rethink for data augmentation](https://zhuanlan.zhihu.com/p/369325673)

And I'm sure 88% is not the best acc. 
For this task is only a benchmark for later work, if time allows  I may try to tune it better.
