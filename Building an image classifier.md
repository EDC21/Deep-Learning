# Building an Image Classifier Using the Sequential API

In this example, we will tackle the dataset,Fashion MNIST.


```python
import tensorflow as tf
from tensorflow import keras

```

## Load the dataset

When loading Fashion MNIST using Keras rather than Sklearn, one important difference is that every image is represented as a 28x28 array rather than a 1D array of size 784. Moreoverm the pixel intensities are represented as integers (from 0 to 255) rather than floats (from 0.0 to 255.0) 


```python
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
```

