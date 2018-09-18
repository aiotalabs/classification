# classification
Repository for emDNN compressed classification models.

## CIFAR-10 classification

* Classification using emDNN compressed ResNet-18 can be done using:
  ```
  python cifar_classifier.py -p models/resnet18/resnet18_emdnn.prototxt -m models/resnet18/resnet18_emdnn.caffemodel
  ```
* To evaluate pretrained model use normalized CIFAR-10 images, using mean = (0.4914, 0.4822, 0.4465) and scaled down by 255.0 and save it to a npz file to be read by [eval_cifar_caffe_model.py](https://github.com/aiotalabs/classification/blob/master/eval_cifar_caffe_model.py) or else download already normalized CIFAR-10 from [here](https://drive.google.com/file/d/1tg834WoaYlzetgcNenuIbiPNZkquJzSt/view?usp=sharing).


