
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
