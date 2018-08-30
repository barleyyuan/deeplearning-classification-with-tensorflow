# deeplearning-classification-with-tensorflow
基于TensorFlow的经典分类网络的实现——vgg16，resnet系列

### 更新说明：添加了基于tf.keras的迁移学习，在keras官方提供的ImageNet上的预训练模型的基础上进行finetune。


## 使用手册
### 环境依赖
Ubuntu16.04 + python3.6.6 + tensorflow1.10.0

### 数据集格式   
dataset-|-class0-|-00001.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-00002.jpg   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-...   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-class1-|-...    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-class2-|-...    
训练集和测试集不需提前划分，修改resnet/train.py中的test_rate参数，可以修改测试集划分比例。

### 训练与预测
#### Tensorflow版本 —— 以ResNet为例
##### 训练
###### 修改train.py中的参数： 
网络结构参数    
`depth`: resnet深度（层数），可选值：18，34（浅层），50，101，152（深层）；   
`num_classes`: 类别数，即网络输出层的神经元个数；   
     
训练参数      
`optimizer`：优化器，可选值：'sgd', 'adam', 'momentum'（还可选用其他优化器，在utils.create_optimizer()函数中添加）；    
`learning_rate`, `momentum`, `batch_size`: 各种训练参数；  
`epochs`：最大迭代次数；   
`epochs_every_test`: 每epochs_every_test个epochs测试一次；   
`epochs_every_save`: 每epochs_every_save个epochs保存一次模型；
`early_stop_num`: 连续early_stop_num个epochs的train_accuracy==1.0且train_loss==0.0，或连续early_stop_num个epochs的val_accuracy和val_loss均没有提升时，提前结束训练（由于我参与的项目的数据较少，故没有设置验证集，之后会来填坑）；    
`method`: 训练方式，可选值：'restart'（从头开始）, 'restore'（继续训练）；   
       
各种路径      
`trained_model_directory`: 已训练模型目录，当训练方式为'restore'时必选；   
`model_directory`: 模型保存路径；  
`model_name`: 模型保存名称；   
`log_directory`: 日志文件和tensorboard文件保存路径，日志中保存了各种网络训练时输出的结果；    
`log_filename`: 日志文件保存名称；   
`summarize`: 是否利用tensorboard记录训练结果的布尔值，可选值：True或False

###### 运行
```
python resnet/train.py
```

##### 预测
###### 修改resnet/predict.py中的参数
model_path: ckpt模型所在路径
###### 运行
```
python predict.py    
$ Input image filename: 输入预测图片的路径
```

#### tensorflow.keras版本 —— 以ResNet50 finetune为例
##### 数据处理
要求数据格式与上述一致，在训练之前，把数据集分割为train, validation, test 三部分
###### 修改 data_split.py 参数
路径        
`main_data_path`: 数据集所在路径
`new_main_path`: 分割后的新数据集所在路径
      
分割参数       
`test_ratio`: 测试集所占比例
`val_ratio`: 验证集所占比例
`seed`: 随机种子

###### 运行
```
python tf_keras/data_split.py
```

##### 训练
训练任务为在keras提供的预训练模型的基础上进行finetune，当在区别于imagenet1000类的新类别上构建分类任务时，推荐这种方法；
注：    
如果希望参数随机初始化，不需finetune时，修改代码第40行为：
```
base_model = ResNet50(weights=None, include_top=False, pooling="avg")

```
如果希望只训练最后用于分类的全连接层，而不训练预训练模型中的其他参数时，把50-51行的注释'#'删除掉：
```
# if train only the top layers, uncomment the following two lines
for layer in base_model.layers:
    layer.trainable = False
```
###### 修改 tf_keras/train_resnet50.py 参数
网络参数
`num_classes`：类别数；     
      
 训练参数
 `batch_size`: 就是batch_size；    
 `epochs`： 最大迭代epochs数；    
 `validation_steps`：验证集生成器的返回次数，若validation没有设置batch，则validation_steps设为1即可；   
 `opt`： 优化器，可选值有：'sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam';    
 `learning_rate`： 学习率；
 `momentum`： 动量，opt为'sgd'时必选；
       
 路径     
 `model_path`： 模型保存路径     
 `data_path`：数据集所在路径       
 
 ###### 运行
 ```
 python tf_keras/train_resnet50.py
 ```
 
##### 预测
###### 修改 tf_keras/predict_resnet50.py 参数
`file_path`： 已训练的h5模型的路径+文件名      
`class_list`: 与数据集中类别名称顺序相一致的类别名称的列表    
###### 运行 
```
python tf_keras/predict_resnet50.py
$ Input image filename: 输入预测图片的路径+文件名
```


