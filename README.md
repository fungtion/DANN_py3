## This is a pytorch implementation of the paper *[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/)*


#### Environment
- Pytorch 1.0
- Python 3.7

#### Network Structure


![p8KTyD.md.jpg](https://s1.ax1x.com/2018/01/12/p8KTyD.md.jpg)

#### Dataset

First, download target dataset mnist_m from [pan.baidu.com](https://pan.baidu.com/s/1pXaMkVsQf_yUT51SeYh27g) fetch code: kjan or [Google Drive](https://drive.google.com/open?id=0B_tExHiYS-0veklUZHFYT19KYjg)

```
cd DANN_py3
mkdir dataset
cd dataset
mkdir mnist_m
cd mnist_m
tar -zvxf mnist_m.tar.gz
mkdir models
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

