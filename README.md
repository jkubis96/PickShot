### PickShot - Python package for CNN-based filtering of low-quality images from analysis

<img src="https://img.shields.io/badge/python-3.12-blue" alt="Python Version"/>

<br />



<p align="right">
<img  src="https://github.com/jkubis96/Logos/blob/main/logos/jbs_current.png?raw=true" alt="drawing" width="180" />
</p>


### Author: Jakub Kubiś

<div align="left">
 Institute of Bioorganic Chemistry<br />
 Polish Academy of Sciences<br />
 Laboratory of Single Cell Analyses<br />

<br />

<p align="left">
<img  src="fig/lsca.png" alt="drawing" width="250" />
</p>
</div>


<br />


## Description

PickShot - Python Package for Automated Image Quality Assessment with CNNs
PickShot is a powerful Python package designed to facilitate the training and deployment of Convolutional Neural Network (CNN) models for automatic quality assessment of microscopy or imaging cytometry data.

This tool allows users to:

* Train custom models on user-selected high-quality and low-quality images.

* Automatically classify and filter images based on quality, streamlining the process of collecting, processing, and analyzing large datasets.

<p align="center">
<img  src="fig/pickshot.png" alt="drawing" width="400" />
</p>
</div>

By accelerating image quality assessment, PickShot significantly enhances the efficiency of comprehensive image analysis workflows.

📂 In the Models tab, ready-to-use pretrained models are available for download. See [saved_models](saved_models/).

📌 Details of usage are provided below.



<br />

#### Table of contents

[Installation](#installation) \
[Usage](#usage)
[Examples](#examples)
1. [Model Training](#mt) \
1.1 [Model parameters](#mp) \
1.1.1 [Adjustment image shape](#mps) \
1.1.2 [Train/Test split](#mptt) \
1.1.3 [Activate function](#af) \
1.1.4 [Epochs](#e) \
1.1.5 [Batch size](#bs) \
1.2 [Training](#tr) \
1.2.1 [Add paths](#aps) \
1.2.2 [Train](#train) \
1.2.3 [Get model stats](#gms) \
1.2.4 [Save model](#svm) 
2. [Prediction](#prd) \
2.1 [Load model and create object](#lmac) \
2.1.1 [From file](#ff) \
2.1.2 [From GitHub](#gf) \
2.2 [Predict](#prdt) 

<br />

<br />

#### Installation <a id="installation"></a>

* Pip:

```
pip install pickshot
```

* Poetry:

```
git clone https://github.com/jkubis96/PickShot.git
cd PickShot
poetry install
```

<br />


### Usage <a id="usage"></a>


#### 1. Model Training <a id="mt"></a>


    Class for training a convolutional neural network model for image classification.

    Attributes:
        _image_shape (tuple): Target shape for input images (default (50, 50)).
        _drop_images (list): List of image paths to be excluded from training.
        _save_images (list): List of image paths to be included in training.
        _train_paths (list): Paths of training images.
        _train_labels (list): Labels for training images.
        _test_paths (list): Paths of testing images.
        _test_labels (list): Labels for testing images.
        test_size_val (float): Proportion of data used for testing (default 0.2).
        activation (str): Activation function used in the model (default 'relu').
        model_storage: The trained model.
        epochs_val (int): Number of training epochs (default 10).
        batch_size_val (int): Batch size used during training (default 32).
        model_stats (dict): Dictionary holding performance metrics.
    

```
from pickshot import TrainingModel

# create instance of the TrainingModel class
model = TrainingModel()
```

<br />

#### 1.1 Model parameters <a id="mp"></a>

##### 1.1.1 Adjustment image shape <a id="mps"></a>

```
model.image_shape
```

        Returns the shape of the input images.


```
model.image_shape(value = (50,50))
```

        Sets the shape of the input images.

        Parameters:
            value (tuple): Shape of the images.

        Raises:
            ValueError: If value is not a tuple.

<br />

##### 1.1.2 Train/Test split <a id="mptt"></a>

```
model.test_size
```

        Returns the test size for splitting data.

```
model.test_size(value = .3)
```

        Sets the test size for splitting data.

        Parameters:
            value (float): Test size as a float.

        Raises:
            ValueError: If value is not a float.
           
<br />

##### 1.1.3 Activate function <a id="af"></a>

```
model.activation_fun
```

        Returns the activation function used in the model.
        

```
model.activation_fun(value = "relu")
```

        Sets the activation function.

        Parameters:
            value (str): Name of the activation function.

        Raises:
            ValueError: If value is not a string.

<br />


##### 1.1.4 Epochs <a id="e"></a>

```
model.epochs
```

        Returns the number of epochs for training.


```
model.epochs(value = 10)
```

        Sets the number of epochs for training.

        Parameters:
            value (int): Number of epochs.

        Raises:
            ValueError: If value is not an integer.


<br />


##### 1.1.5 Batch size <a id="bs"></a>

```
model.batch_size
```

        
        Returns the batch size used during training.
        

```
model.batch_size(value = 32)
```
        
        Sets the batch size for training.

        Parameters:
            value (int): Batch size.

        Raises:
            ValueError: If value is not an integer.
        

<br />


#### 1.2 Training <a id="tr"></a>

##### 1.2.1 Add paths <a id="aps"></a>


```
model.images_paths(images_to_drop: list, images_to_save: list)
```

        
        Validates and sets the paths for images to be dropped and saved for training.

        Parameters:
            images_to_drop (list): List of image paths to exclude.
            images_to_save (list): List of image paths to include.
        
<br />


##### 1.2.2 Train <a id="train"></a>


```
model.train()
```

    
        Runs the entire model training process.

<br />

##### 1.2.3 Get model stats <a id="gms"></a>


```
model.get_notes()
```

        Prints and returns the model's performance metrics.

        Returns:
            dict: A dictionary of the model's performance metrics (Accuracy, Precision, Recall, F1-Score).
                 

<br />

##### 1.2.4 Save model <a id="svm"></a>


```
model.save_model(name: str, path=os.getcwd())
```

        Saves the trained model to a file.

        Parameters:
            name (str): The name for the saved model file.
            path (str): Directory path to save the model (defaults to current working directory).
        

<br />


#### 2. Prediction <a id="prd"></a>


    
    Class for processing images and making predictions using a loaded model.

    Attributes:
        name (str): The name of the model.
        model_storage: The loaded model used for predictions.
        _image_shape (tuple): The target shape to which images will be resized before prediction.
    

```
from pickshot import PickShot
```
<br />

#### 2.1 Load model and create object <a id="lmac"></a>

##### 2.1.1 From file <a id="ff"></a>

```
model = PickShot.load(file_path)
```

        Loads a model from a file.

        Parameters:
            file_path (str): The path to the file containing the model.

        Returns:
            PickShot: A new PickShot object with the loaded model.

        Exceptions:
            FileNotFoundError: Raised if the file cannot be found at the specified path.
        
<br />

##### 2.1.2 From GitHub <a id="gf"></a>

```
model = PickShot.download(url: str, path_to_save=None)
```

        Downloads a model from the internet and saves it to disk.

        Parameters:
            url (str): The URL from which the model will be downloaded.
            path_to_save (str, optional): The directory where the file will be saved (defaults to the current working directory).

        Returns:
            PickShot: A new PickShot object with the downloaded model.

        Exceptions:
            requests.exceptions.RequestException: Raised if an error occurs while downloading the file.
        
<br />

#### 2.2 Predict <a id="prdt"></a>


```
model.predict(path_to_images: str, ident_part: str, pred_value: float = 0.7)

```

                
        Makes predictions for images in a given directory based on a part of their filenames.

        Parameters:
            path_to_images (str): The path to the directory containing the images.
            ident_part (str): A part of the filename used to identify relevant images.
            pred_value (float, optional): The threshold for the prediction (default is 0.7).

        Returns:
            dict: A dictionary containing image IDs and their corresponding predictions ("pass" or "drop").
        
        

<br />

#### Examples <a id="examples"></a>

* training:

```
from pickshot import TrainingModel

# create instance of the TrainingModel class
model = TrainingModel()

# paths to images
images_to_drop = ['img_1.tif' , 'img_2.tif', 'img_3.tif']
images_to_save = ['img_4.tif' , 'img_5.tif', 'img_6.tif']

# put paths into model
model.images_paths(images_to_drop, images_to_save)

# train
model.train()

# get model stats (accuracy, precission, recall, F1-score)
model.get_notes()
```

<p align="center">
<img  src="fig/stats.png" alt="drawing" width="150" />
</p>
</div>

```
# save model
model.save_model(name = 'test_model', path = os.getcwd())
```

* prediction:

```
from pickshot import TrainingModel

# create instance of the PickShot class with model

# from file (eg . own model, previously downloaded model)
model = PickShot.load('test_model_CNN_(50,50).h5')

# ready models from GitHub or other source
url = 'https://github.com/jkubis96/PickShot/raw/refs/heads/draq5_nuclei_model/saved_models/nuclei_CNN_(50,50).h5'

model = PickShot.download(url)

# paths to images for prediction
path_to_images = ['img_1.tif' , 'img_2.tif', 'img_3.tif', 
                  'img_4.tif' , 'img_5.tif', 'img_6.tif']

results_dictionary = model.predict(path_to_images, 
                ident_part = 'CH11.om', 
                pred_value = 0.7)
```


<p align="center">
<img  src="fig/dict.png" alt="drawing" width="150" />
</p>
</div>


<br />


#### Have fun JBS©