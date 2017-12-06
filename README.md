# MethodsofTrainingConvolutionNeuralNetwork
Efficient Methods of Training Convolution Neural Network

## about

**Network(Alexnet)**

* 8 layers
  * 5 convolutional 
  * 3 fully-connected
* ReLU Nonlinearity
* Training on Multiple GPUs(50X than cpu using GTX 580 x2) 
* Local Response Normalization(internal covariance)
![](https://sushscience.files.wordpress.com/2016/12/alexnet2.jpg?w=900)

**Network(VGG)**

*To this end, we fix other parameters of the architecture, and steadily **increase the depth of the network** by adding more convolutional layers, 
which is feasible due to the use of **very small ( 3 × 3) convolution filters** in all layers.*
![](http://cfile4.uf.tistory.com/image/24345341583ED6B718D609)

**Network(inception)**

*Our results seem to yield a solid evidence that approximating the expected optimal **sparse** structure by readily available dense building blocks 
is a viable method for improving neural networks for computer vision.  -Going Deeper with Convolutions(inception)*

* The Brain is sparsely connected(not fully connected)
* dense calculation but sparse network

![](https://hackathonprojects.files.wordpress.com/2016/09/inception_implement.png?w=649&h=337)

**1x1 convolution**

* 1x1 convolution leads to dimension reductionality!!
![](https://i.ytimg.com/vi/rWbz33rMfMQ/maxresdefault.jpg)

**Network(mobilenet)**

* Unstructured sparse matrix operations are not typically faster than dense matrix operations until a very high level of sparsity. but 1x1 conv is general matrix multiply (GEMM) functions.
* F2NK2M to F2NM+ F2NK2. It is reduced to about ⅛  ~ 1/9  of formal method because reduction of computation is 1/N  +1/Dk2, usually M>>K2(eg.K=3,M>=32).
* It reduced the proportion of parameters present in the FC.

![](https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/Reading_Note_20170719_MobileNet_0.png)

**Network(Resnet)**

* residual connection
  * Learning the difference between input and output.
![](https://image.slidesharecdn.com/mrn-161128091530/95/multimodal-residual-learning-for-visual-qa-14-638.jpg?cb=1480324582)

## requirement

* tensorflow-gpu (ver.1.3.1)
* jupyter

### tensorboard

```
tensorboard --logdir /path/to 
```

### train

example using 4 gpu momentum optimizer lr 0.01 imagenet dateaset
```
python3 train_image_classifier.py --num_readers=6 --num_preprocessing_threads=6 --optimizer=momentum --learning_rate=0.01 --num_clones=4 --batch_size=12 --dataset_name=imagenet --dataset_split_name=train --dataset_dir=/home/jhl/data/imagenet/dataset/ --model_name=alexnet_v2 --max_number_of_steps=1000000 --train_dir=/home/jhl/train/imagenet_alexnet_momentum_1000000_step_lr_0.01 --preprocessing_name=alexnet && python3 train_image_classifier.py --num_readers=6 --num_preprocessing_threads=6 --optimizer=adam --learning_rate=0.01 --num_clones=4 --batch_size=12 --dataset_name=imagenet --dataset_split_name=train --dataset_dir=/home/jhl/data/imagenet/dataset/ --model_name=alexnet_v2 --max_number_of_steps=1000000 --train_dir=/home/jhl/train/imagenet_alexnet_adam_1000000_step_lr_0.01 --preprocessing_name=alexnet
```

### inference

```
python3 eval_image_classifier.py --batch_size=100 --num_preprocessing_threads=12 --eval_dir=path/to --checkpoint_path=path/to --dataset_dir=path/to --dataset_name=imagenet --dataset_split_name=validation --model_name=mobilenet_v1 --eval_image_size=128
```

### jupyter


Downloading and converting to TFRecord format **Cifar10** dataset

```
python download_and_convert_data.py \
    --dataset_name=cifar10 \
    --dataset_dir="${DATA_DIR}"
```

## [LICENSE](LICENSE)
