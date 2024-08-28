# transformer 源码实现

```
注意：在3050，4080显卡上测试成功，不确定是否能在CPU上使用，因为用到了第三方训练包
```

### 创建虚拟环境
目前最新版本3.12，我这里选择3.11，向前选择一个版本
```
conda create -n trans python==3.11
```

### 安装 torch
自己选择合适的torch，cuda版本，最好向前选择一个版本，目前最新torch是2.4，所以我选择了2.3
```
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 安装包
下面是我不断试错，总结出来需要安装的包
```
pip install pandas torchkeras torchmetrics ipython tqdm requests accelerate ipywidgets matplotlib
```

### 训练
训练完成，会在当前目录生成两个文件，checkpoint.pth，history.png，一个权重文件，一个训练进度图
```
python trainer.py
```

### 测试
```
(trans) I:\transformer>python test.py
input:
646487066023+29233095105329712497

ground truth:
29233095751816778520

prediction:
29233095751816778520
```