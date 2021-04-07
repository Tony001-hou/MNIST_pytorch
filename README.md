# MNIST Pytorch

This report provides a CNN model with Pytorch, to train and test on MNIST data of handwritten digits, as well as test on single image input.
This report can be opened in Windows OS and run on local devide. 


### Prerequisites

1. Download the google drive folder **mnist**, the google drive will automaticly compress the folder as a zip file and rename the zip file.
2. After downloading, change the zip file name to **mnist**.
3. Uncompress the mnist.zip file and you will get a **mnist** folder.  Assume your mnist folder is under **This PC > Downloads**
4. Open **Anaconda Prompt**. It is recommended to use **Anaconda Prompt** to run this report, because it provides environment to run python files.
5. Please type in the following cammands to change to your folder path. 

```
cd Downloads\mnist
```

6. Please type in the following cammand to install the prerequisite package

```
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

### Traing 
```
python train.py
```

This file will train a cnn model, the model will be saved as **mnist_cnn.pt**

When you want to change the 'data' folder, please open **train.py** and change **line 61** and **line 63**.
```
61  dataset1 = datasets.MNIST('data', train=True, download=False,
62                     transform=transform)
63  dataset2 = datasets.MNIST('data', train=False,
64                     transform=transform)
```

### Test on single image
#### 1. First way
```
python test.py
```
You can download the **mnist_png.rar** from the google drive, this folder contains all the image data of .png format. Then you can copy an image from **mnist_png** folder and put it under **mnist** folder

Then, change the following line 49 in **test.py** with the new image files.

```
49 image = Image.open('25.png')
```

#### 1. First way
```
python test.py
```
You can download the **mnist_png.rar** from the google drive, this folder contains all the image data of .png format. Then you can copy an image from **mnist_png** folder and put it under **mnist** folder

Then, change the following line 49 in **test.py** with the new image files.

```
49 image = Image.open('25.png')
```


#### 2. Second way
```
python test_with_image.py --img_dir 25.png
```
You can change this **25.png** with another file to test on another image





