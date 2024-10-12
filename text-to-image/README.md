# 文生图安装部署

使用两个开源的大模型实现文生图，一个翻译大模型，一个文生图大模型

![样例](../images/exp.png)

### 安装 torch
自己选择合适的torch，cuda版本，最好向前选择一个版本，目前最新torch是2.4，所以我选择了2.3
```
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 安装包
```
pip install -r requirements.txt
```

### 下载文件
翻译大模型
```
https://huggingface.co/Helsinki-NLP/opus-mt-zh-en
```

文生图大模型
```
https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell
```

### 测试
安装完后，就可以测试了
```
(trans) I:\transformer\text-to-image>python main.py
```

### 浏览器中打开
```
http://localhost:7860/
```