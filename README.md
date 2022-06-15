## This is a pytorch implementation of the paper *[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/)*


#### Environment
- Pytorch 1.6
- Python 3.8.5

#### Network Structure


![p8KTyD.md.jpg](https://s1.ax1x.com/2018/01/12/p8KTyD.md.jpg)

#### Dataset

First, download target dataset mnist_m from [pan.quark.com](https://pan.quark.cn/s/f4002a4fadbc) or [Google Drive](https://drive.google.com/open?id=0B_tExHiYS-0veklUZHFYT19KYjg), and put mnist_m dataset into dataset/mnist_m, the structure is as follows:

```
--dataset--mnist_m--mnist_m_train
                 |--mnist_m_test
                 |--mnist_m_train_labels.txt
                 |--mnist_m_test_labels.txt
                 |--.gitkeep

```

#### Training

Then, run `python main.py`


#### Docker

- build image

```bash
docker build -t pytorch_dann .
```

- run docker container

```bash
docker run -it --runtime=nvidia \
  -u $(id -u):$(id -g) \
  -v /YOUR/DANN/PROJECT/dataset:/DANN/dataset \
  -v /YOUR/DANN/PROJECT/models:/DANN/models \
  pytorch_dann:latest \
  python main.py

```

