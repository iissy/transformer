# transformer 源码实现

注意：在3050，4080显卡上测试成功，在老旧电脑（7代i7，8G内存，集显）上使用CPU训练，需要4个小时，下图为训练精度变化情况，经过几轮训练，精度会迅速爬升。

![精度](./images/history.png)

### 创建虚拟环境
目前最新版本3.12，我这里选择3.11，向前选择一个版本
```
conda create -n trans python==3.11
conda activate trans
```

### 安装 torch
自己选择合适的torch，cuda版本，最好向前选择一个版本，目前最新torch是2.4，所以我选择了2.3
```
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 安装包
下面是我不断试错，总结出来需要安装的包
```
pip install -r requirements.txt
```

### 测试
安装完后，你可以直接用我已经训练好的权重测试
```
(trans) I:\transformer>python inter.py
input:
646487066023+29233095105329712497

ground truth:
29233095751816778520

prediction:
29233095751816778520
```

### 训练
开始自己训练，会在当前目录生成checkpoint.pth权重文件，如果想更快训练完成，可以修改训练轮数。
```
python trainer.py
```
你也可以使用集成了第三方包的训练，效果是一样的，额外会生成一个损失，精度图
```
python keras_trainer.py
```

## 参考
https://github.com/lyhue1991/torchkeras