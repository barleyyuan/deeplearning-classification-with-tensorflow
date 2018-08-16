# deeplearning-classification-with-tensorflow
基于TensorFlow的经典分类网络的实现——vgg16，resnet系列

## 使用手册——以resnet为例
### 环境
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
### 修改train.py中的参数： 
####  网络结构参数
depth: resnet深度（层数），可选值：18，34（浅层），50，101，152（深层）；   
num_classes: 类别数，即网络输出层的神经元个数；   
#### 训练参数
optimizer：优化器，可选值：'sgd', 'adam', 'momentum'（还可选用其他优化器，在utils.create_optimizer()函数中添加）；    
learning_rate, momentum, batch_size: 各种训练参数；  
epochs：最大迭代次数；   
epochs_every_test: 每epochs_every_test个epochs测试一次；   
epochs_every_save: 每epochs_every_save个epochs保存一次模型；
early_stop_num: 连续early_stop_num个epochs的train_accuracy==1.0且train_loss==0.0，或连续early_stop_num个epochs的val_accuracy和val_loss均没有提升时，提前结束训练（由于我参与的项目的数据较少，故没有设置验证集，之后会来填坑）；    
method: 训练方式，可选值：'restart'（从头开始）, 'restore'（继续训练）；   
#### 各种路径
trained_model_directory: 已训练模型目录，当训练方式为'restore'时必选；   
model_directory: 模型保存路径；  
model_name: 模型保存名称；   
log_directory: 日志文件和tensorboard文件保存路径，日志中保存了各种网络训练时输出的结果；    
log_filename: 日志文件保存名称；   
summarize: 是否利用tensorboard记录训练结果的布尔值，可选值：True或False
### 运行
python resnet/train.py    
