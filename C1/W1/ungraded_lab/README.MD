# [Coursera] C1W1 Introduction to TensorFlow , Machine Learning , DeepLearning - A new programming paradigm

> **Goal**
* 집 값 예측의 정확도 모니터
* single layer neural network 로 부터 오는 집 값 예측 분석
* TensorFlow 를 사용하여 linear model 을 fitting 하기 위한 signle layer neural network 구성 


```batch
< 참고 > 

해당 course 의 notebook 들은 colab 환경에서 실행 가능   
local 에서 실행하고 싶다면 아래 repository 를 clone 및 실행 환경 구성 
* git clone https://github.com/https-deeplearning-ai/tensorflow-1-public

tensorflow==2.7.0
scikit-learn==1.0.1
pandas==1.1.5
matplotlib==3.2.2
seaborn==0.11.2

```

## A new programming paradigm

### L3 The 'Hello World' of neural networks

![](https://velog.velcdn.com/images/hobbang2/post/38bf02c8-3cdd-4088-a6e0-0cf593cc103b/image.png)

전통적인 프로그래밍에서 개발자는 규칙을 세분화하여 각 규칙에 맞는 코드를 개발했다.   
새로운 패러다임에서는 데이터 라벨링을 통해 컴퓨터가 규칙을 찾아낸다. 

> 수학과 통계적 지식은 모델을 최적화하는 데에 도움이 된다.   
> keras 와 TensorFlow 는 해당 지식들이 함수로 이미 작성되어 있다는 장점이 있다. 

```python
# 1 Dense layer and 1 unit : single neural 
model = keras.Sequential( [keras.layers.Dense( units=1 , input_shape = [1])])
# sgd : stands for stochastic gradient descent
model.compile( optimizer ='sgd' , loss ='mean_squared_error')

# data 표현하기 
xs = np.array([ -1.0 , 0.0 , 1.0 , 2.0 , 3.0 , 4.0 ] , dtype = float ) 
ys = np.array([ -3.0 , -1.0 , 1.0 , 3.0 , 5.0 , 7.0 ] , dtype = float ) 

# train set 을 500 번 반복 
model.fit( xs, ys , epochs = 500 ) 

print( model.predict([10.0]) ) 
```

**loss function** and **optimizer**
- 신경망은 x 와 y의 관계에 대해 아무런 정보가 없다. 
- 적절한 식을 추측하고 loss function 으로 평가 후 optimizer 가 다음 추측을 할 수 있도록 데이터를 전달해준다. `convergence`

**model.predict([10.0]) 은 19에 근사한 값을 출력한다.**
- 왜 19가 아니라 19에 근사한 값을 출력할까 ?  
	i ) 적은 데이터로 훈련되었음  
	ii ) neural network 는 확률에 따라 y 값을 추론함  
    
### L4 Working through 'Hello World' in TensorFlow and Python
https://github.com/hobbang2/TensorFlowCertification/blob/main/C1/W1/ungraded_lab/C1_W1_Lab_1_hello_world_nn.ipynb

간단한 코드 작성을 통해 neural network 의 동작을 확인해보자 ! 

#### library import 
```python
import tensorflow as tf
import numpy as np
# sequence model 을 작성하기 위한 keras
from tensorflow import keras 

print( tf.__version__ ) 
```

#### single neural network
neural network 는 optimizer 와 loss function 을 통해 학습한다.  
epoch 만큼 반복하면서 추측 값을 평가하고 optimizer 를 통해 더 나은 추론을 한다. 
```python
# Build a simple Sequential model
model = tf.keras.Sequential([keras.layers.Dense(units=1 , input_shape = [1] )])

# Compile the model
model.compile( optimizer='sgd' , loss = 'mean_squared_error' )
```

#### Providing the Data 
`y=2x-1` 관계에 있는 x 와 y 값을 입력으로 제공해보자. 
각 값은 numpy array 로 전달할 수 있다. 
```python
xs = np.array([ -1.0 , 0.0 , 1.0 , 2.0 , 3.0 , 4.0 ] , dtype = float ) 
ys = np.array([ -3.0 , -1.0 , 1.0 , 3.0 , 5.0 , 7.0 ] , dtype = float ) 
```

#### Training the Neural Network
`model.fit()` 을 통해 neural network 를 훈련시킬 수 있다. 
각 loop 를 진행하면서 loss 값을 평가하고 optimizer 로 새로운 추론을 찾는다. 
```python
model.fit( xs , ys , epochs = 500 )
```
![](https://velog.velcdn.com/images/hobbang2/post/c458e0c3-704e-4244-a93c-55cadfa39a9e/image.png)

#### Prediction
입력 데이터 양이 적은데다 , neural network 는 확률적 추론을 하기 때문에 정확히 19를 출력하지는 않는다. 

> memo : neural network 는 확률적 추론값을 도출한다.
```python
print(model.predict([10.0])
# [[18.979267]]
```

### Quiz

7. What does model.fit do ? 
> answer : It trains the neural network to fit one set of values to another

### C1W1_Assignment : Week 1 Assignment : Housing Prices
house 의 기본 대여 가격은 50k 이고 , bed 한 개당 50k 이다. 
- 1 bedroom : 100k , 2 bedroom : 150k , ...
> memo  
> 400k 처럼 큰 값을 y value 로 줄 필요는 없다.    
> y 값의 scale 을 낮추면 network가 더 잘 작동할 것이다. 
