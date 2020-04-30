[toc]

## 一、runx介绍

**runx** [origin repo](https://github.com/NVIDIA/runx) 是NVIDIA官方刚放出来的一个深度学习实验管理工具，可以在做实验的同时很方便地实现下列一些常用的功能：

* 超参数、计算资源等的配置和记录
* 实验过程、实验结果、日志等的记录
* 实验总结


通过阅读源码和自己使用了两天以后，我认为runx是目前我见过的最简洁和易用的实验管理工具了，其优点我总结如下：
1. 几乎能覆盖一个完整实验管理需要的所有功能：超参和硬件资源设置、实验过程和结果记录、实验总结；
2. runx的源码很简单，其实就是在argparse、tensorboard等基础上进行了一层封装，因此可以很方便地嵌入自己的代码中（比自己造的轮子要完善），同时也能针对自己的需求对源码进行功能上的定制；
3. runx利用文件夹对不同实验进行记录，很方便对代码改动进行管理，不会出现改动代码以后找不回原来某些实验设置的痛苦。

### example
举一个简单的例子来介绍runx，例如你存在一个如下的项目：
```
> python train.py --lr 0.01 --solver sgd
```

如果需要对lr=[0.01, 0.02]和solver=[sgd, adam]进行实验，那么需要写四次不同的实验设置代码或者脚本，但是通过使用runx，只需要
在项目中添加一个yaml文件（e.g. sweep.yml）:
```
cmd: 'python train.py'

hparams:
  lr: [0.01, 0.02]
  solver: ['sgd', 'adam']
```
然后运行
```
> python -m runx.runx sweep.yml

python train.py --lr 0.01 --solver sgd
python train.py --lr 0.01 --solver adam
python train.py --lr 0.02 --solver sgd
python train.py --lr 0.02 --solver adam
```
runx会自动计算参数值的叉乘，然后批量产生四个runs。
runx也能很好的支持farm，在项目中新建一个名为.runx的文件：
```
LOGROOT: /home/logs
FARM: bigfarm

bigfarm:
  SUBMIT_CMD: 'submit_job'
  RESOURCES:
     gpu: 2
     cpu: 16
     mem: 128
```
现在可以使用定义的bigfarm来运行：
```
> python -m runx.runx sweep.yml

submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.01 --solver sgd"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.01 --solver adam"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.02 --solver sgd"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.02 --solver adam"
```
这里的submit_job是farm提交命令的占位符，比如brain++上这里可以改为rlaunch。
runx会把每一个run的结果记录在不同的目录里，只需要在yaml文件中增加一个logdir字段：
```
CMD: 'python train.py'

HPARAMS:
  lr: [0.01, 0.02]
  solver: ['sgd', 'adam']
  logdir: LOGDIR
```
现在launch这些jobs时，runx会自动生成不同的目录并传给训练脚本：
```
> python -m runx.runx sweep.yml

submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.01 --solver sgd  --logdir /home/logs/athletic-wallaby_2020.02.06_14.19"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.01 --solver adam  --logdir /home/logs/industrious-chicken_2020.02.06_14.19"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.02 --solver sgd  --logdir /home/logs/arrogant-buffalo_2020.02.06_14.19"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.02 --solver adam  --logdir /home/logs/vengeful-jaguar_2020.02.06_14.19"
```
实验结束后，可以利用`sumx`来对实验结果进行一个简单的总结分析：
```
> python -m runx.sumx sweep --sortwith acc

        lr    solver  acc   epoch  epoch_time
------	----  ------  ----  -----  ----------
run4    0.02  adam    99.1   10     5:11
run3    0.02  sgd     99.0   10     5:05
run1    0.01  sgd     98.2   10     5:15
run2    0.01  adam    98.1   10     5:12
```


## 二、runx模块和使用
runx一共有三个模块，分别是：
* runx：用一个yaml格式的文件设置各种参数，每个参数可以有多个值，运行时计算不同值的叉乘，产生对应的命令，同时每一个运行的训练命令会新建一个输出文件并把对应的代码复制过去；
* logx：记录metrics、中间信息、checkpoints和tensorboard；
* sumx：总结不同组实验训练和测试的结果。

runx的安装很简单，可以直接用pip：
```
> pip install runx
```
或者从源码安装：
```
> git clone https://github.com/NVIDIA/runx
> cd runx
> python setup.py install --user
```
或者把runx作为项目的一个子模块：
```
> cd <your repo>
> git submodule add -b master ssh://git@gitlab-master.nvidia.com:runx.git
```


### runx
在项目中新建一个`.runx`文件，`.runx`是一个项目配置文件，定义了如何向你的集群中提交任务和把输出放在哪里。

`.runx`中有如下参数：
* LOGROOT - 存放logs的根目录
* FARM - 使用对应的farm提交job或者交互式地运行
* SUBMIT_CMD - 集群提交命令
* RESOURCES - 超参数会被传递给submit_cmd
* CODE\_IGNORE\_PATTERNS - 把代码复制到输出目录时忽略这些格式的文件

下面是一个`.runx`文件的例子：
```
LOGROOT: /home/logs
FARM: bigfarm
CODE_IGNORE_PATTERNS: '.git,*.pyc,docs*,test*'

# Farm resource needs
bigfarm:
    SUBMIT_CMD: 'submit_job'
    RESOURCES:
        image: mydocker-image-big:1.0
        gpu: 8
        cpu: 64
        mem: 450

smallfarm:
    SUBMIT_CMD: 'submit_small'
    RESOURCES:
        image: mydocker-image-small:1.2
        gpu: 4
        cpu: 32
        mem: 256
```

runx有两个实验层级：experiments和runs。一个experiments对应一个yaml文件，并且可能拥有许多runs（超参数叉乘的组合）。

runx会新建一个父级目录并且为每一个run新建一个目录，父级目录是`LOGROOT/<experiment name>`，例子中sweep.yml的experiment name就是 sweep。
```
/home/logs
  sweep/
     curious-rattlesnake_2020.02.06_14.19/
     ambitious-lobster_2020.02.06_14.19/
     ...
```
每一个单独run的目录名由一个随机的`coolname`和日期时间组成，名字也可以通过yaml文件里的RUNX.TAG设置。

runx会把代码复制到每一个run里，因此runx（1）允许运行某些runs以后更改源代码，（2）能够准确记录特定run使用的代码从而避免混乱（这里需要提醒的是一定要把代码和data分开，否则会产生大量冗余copy）。

一个runx yaml文件的参数设置如下所示：
```
HPARAMS: [
  {
   logdir: LOGDIR,
   adam: true,
   arch: alexnet,
   lr: [0.01, 0.02],
   epochs: 10,
   RUNX.TAG: 'alexnet',
  },
  {
   arch: resnet50,
   lr: [0.002, 0.005],
   RUNX.TAG: 'resnet50',
  }
]
```
这里的list中包含两个dict项，对应了两组实验超惨设置：
1. arch = alexnet with lr=[0.01, 0.02]
2. arch = resnet50 with lr=[0.002, 0.005]
这里的一个优点是第二个dict中未写出的项会继承第一个dict中的对应项，避免输入冗余设置，也能很方便地看出来改动了哪些参数。把上述yaml提交到runx后，会得到如下四组runs：
```
submit_job ... --name alexnet_2020.02.06_6.32  -c "python train.py --logdir ... --lr 0.01 --adam --arch alexnet --epochs 10
submit_job ... --name alexnet_2020.02.06_6.40  -c "python train.py --logdir ... --lr 0.02 --adam --arch alexnet --epochs 10
submit_job ... --name resnet50_2020.02.06_6.45 -c "python train.py --logdir ... --lr 0.002 --adam --arch resnet50 --epochs 10
submit_job ... --name resnet50_2020.02.06_6.50 -c "python train.py --logdir ... --lr 0.005 --adam --arch resnet50 --epochs 10
```

### logx
logx用于记录实验过程输出的信息和checkpoints，对于checkpoints，logx会记录最新的和最好的，从而节省磁盘空间。

logx的使用很简单，也能很方便地把print和tensorboard的代码改成logx。
首先在代码里导入logx：
```
from runx.logx import logx
```
在使用logx时，必须先初始化：
```
logx.initialize(logdir=args.logdir, coolname=True, tensorboard=True)
```
然后按照如下修改原先的代码就ok了：

| From                | To                | What                      |
| ------------------- | ----------------- | ------------------------- |
| print()             | logx.msg()        | stdout messages           |
| writer.add_scalar() | logx.add_scalar() | tensorboard scalar writes |
| writer.add_image()  | logx.add_image()  | tensorboard image writes  |
|                     | logx.save_model() | save latest/best models   |

为了sumx能够读取run的结果，需要在logx中记录metrics：
```
# define which metrics to record
metrics = {'loss': test_loss, 'accuracy': accuracy}
# push the metrics to logfile
logx.metric(phase='val', metrics=metrics, epoch=epoch)
```
* `phase`表示metric是训练还是验证阶段的
* 对于验证阶段的metric，需要设置 idx == epoch，对于训练阶段，idx就是迭代的计数

最后，logx也能用于保存模型和checkpoints（最新和最好的）。
```
save_dict = {'epoch': epoch + 1,
             'arch': args.arch,
             'state_dict': model.state_dict(),
             'best_acc1': best_acc1,
             'optimizer' : optimizer.state_dict()}
logx.save_model(save_dict, metric=accuracy, epoch=epoch, higher_better=True)
```
注意，这里需要人为指定metric是越小越好还是越大越好。

### sumx
sumx可以快速的对实验结果做个总览：
```
> python -m runx.sumx sweep
        lr    solver  acc   epoch  epoch_time
run4    0.02  adam    99.1  10     5:21
run3    0.02  sgd     99.0  10     5:02
run1    0.01  sgd     98.2  10     5:40
run2    0.01  adam    98.1  10     5:25
```
* 可以用`--sortwith`来输出特定的排序（比如根据accuracy）
* sumx会输出run当前的epoch
* epoch_time表示平均每个epoch的耗时
* 可以用`--ignore`来限制哪些项应该打印

## 三、如何在brain++上使用runx
我根据rlaunch的命令格式把runx的源码改了一下，经测试可以在brain++上使用。直接把源码down下来按照上面的源码安装就OK了。安装以后可以利用源码中的example测试：

**1. runx**

把.runx修改为如下格式,logroot设置成你自己的路径，submit_cmd改成rlaunch：
```
LOGROOT: /data/log/nndistortion
FARM: bigfarm

bigfarm:
  SUBMIT_CMD: 'rlaunch'
  RESOURCES:
     gpu: 2
     cpu: 16
     memory: 20000
```
mnist.yml修改为如下（去掉原来的cd）：
```
CMD: python mnist.py

HPARAMS:
   lr: [0.01, 0.02]
   momentum: [0.5， 0.25]
   logdir: LOGDIR
```
其他的tag可以按照第二部分的模块使用说明设置（注意把代码里的data路径改成自己的，我建议把data和code分开，否则每一次都会copy data），然后运行
```
python -m runx.runx mnist.yml
```
即可。

**2. logx**

logx直接看example里的训练代码。

**3. sumx**

实验运行完成以后运行：
```
python -m runx.sumx mnist
```
就可以得到一个简单的实验结果总结。

> 这里需要指出的是，一组farm没有并发执行，但是可以根据自己的参数多设置几组farm，一次运行多个yaml文件就行。后续有时间可以尝试加入多线程并发执行。