This report provides a CNN model with Pytorch, to train and test on MNIST data of handwritten digits, as well as test on single image input.
This report can be opened in Windows OS and run on local devide. 


### Prerequisites

Please place the **data** folder and other files under at same location

It is recommended to use **Anaconda Prompt** to run this report, because it provides environment to run python files.
Please type in the following cammand to install the prerequisite package

```
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

### Traing 
```
python train.py
```

This file will train a cnn model, the model will be saved as **mnist_cnn.pt**


### Test on single image
```
python test.py
```

You can copy an image from mnist_png folder and put it at the same direction as the train.py and test.py and change the following code in **test.py** with the new image files.

```
image = Image.open('25.png')
```





