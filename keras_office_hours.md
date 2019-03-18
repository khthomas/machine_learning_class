# ML Office House
* Keras is a high level library that sits on top of other deep learning models.
* Keras is supported on CPU, GPU, and TPU (A Tensor Processing Unit).
* A TPU is designed to do tensor calculations. Made to run NNets very efficently (fit and run)
* It sits on top of several backends (tensorflow, CNTK, MXNet, and Theano).


## TPUs
Specific hardware designed to do tensor calculations. The types of tensors that you use depends on the networks that you use. You can get clusters of them on the cloud (Google Cloud) to train models ("Cloud TPUs"). 

## Keras Sequential API
* Allows us to build models very quickly. The idea is that layers are built sequentially (like legos). Can also build out OOP style.

## What A Tensor
* In deep learning, data is stored in tensors
* A tensor depends on the dimensions of the tensor
    * Scalar = 0D
    * Vector = 1D
    * Matrix = 2D tensor
    * Multidimension = 3D or greater

## Neural Networks
* MLPs - Multi Layer Perceptrons - traditional and fully connected (or feed forward). Typically linear and are not good for sequential or time series data.
* RNN - Recurrant NNet 
* CNN - Convultional Neural Networks - good for multidimensional data (images and video). Extracts feature maps.
* Often times MLPs, RNN, and CNN are combined together for deep learning.
* GAN - Generative Adversarial Networks. Two networkds. A generator that creates data and a discriminator that determins if the data real or generated (fake). Once the discriminator can no longer see the difference between real and generated data it is discarded and the generator is used to create realistic data. 

## Working with Raw Tensorflow
Its kind of a pain in the ass (Keras makes it easy)
```{python}
message = tf.constant("Hello World")
with tf.Session() as session:
    session.run(message)
    print(message.eval())
    
    #check for devices
    devices = session.list_devices()
    for d in devices:
        print(f"\n {d}")

```

## Keras Validation
```{python}
K.epsilon()d
```

## Regularizaiton, Activation Functions, and Optimizers
**Regularization** 
Regularization - Used to prevent overfitting. In Keras, bias, weights, and activation ouput can be regularized by a layer (NNets with smaller parameter values are more insensitive to noise in the input data). 
    * L1 - Penalty function - favor smaller variable values using the given penalty function. Uses a fraction of the sum of the absolute variable values. Good for variable selection.
    * L2 - Penalty function. Better for overfitting. 
    * Drop out - designate a percentage and drops out that percentage.
        EG: 45% drop out (1 - 0.45) = 65% of the neurons or hidden units that participate for the next hidden. They are dropped randomly. NNets typically want to over fit, so we want to increase the robustness of the model. 
    
**Activation Functions**
A weighted sum. Squash values to a specific range.
* Logistic Sigmoid (0 to 1)
* TanH - ( -1 to 1) (matters when you have negative values)
* ReLU - Goes from zero to the maximum value of your data. Important for Deep networks due to the vanishing gradient problem (eventually your gradients vanish befor you update all of your weight starting at the end and chain rule back, as you have more layers the gradient gets smaller and smaller)

**Optimizers**
* Optimizers have the objective of minimizng the loss function. 
* Loss is the default evalution metic in Keras, but you can choose others
* Each is tunable
* Learning rate - controls how much we adjust weights in our nnet relative to the loss gradient. If to great there is a difficulty finding the global minimum. Lower values tend to be better.
* Momentum - is used only with the gradient to accumulate the gradient of the past steps to detemine the direction to go. 
* Examples: Stochastic gradient descent, ADAM (combines momentum on RMS Prop), RMS Prop (propegation) (tries to dampen osccilations during gradient descent, but not only by momentum but also with learning rate), 












