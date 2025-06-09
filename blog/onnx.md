2023-10-07

# What is ONNX? Why Should You Learn It?

The most practical way to deploy deep learning models everywhere.

> You can access all the code from this article in my [GitHub repository](https://github.com/cobanov/onnx-examples).

## Who Should Read This Article

Learning to use ONNX will definitely come in handy if you're working on machine learning, deep learning, or AI projects. With AI creeping into even the simplest applications these days, if you're curious about how to leverage it when the need arises, or if any of the situations below apply to you, I'd recommend giving this a read.

- If you have limited knowledge about deep learning models but you're a web or mobile developer
- If you need to run your AI models on different types of devices
- If you want to optimize your existing models
- If you need to quickly deploy pre-trained AI models
- If you're forced to use multiple deep learning frameworks in the same application
- If you need to create a model in one programming language and then deploy it in a completely different environment (for example, creating a model in Python but deploying it in C# or JavaScript)
- Probably other scenarios I haven't thought of that you'll find in the [documentation](https://onnxruntime.ai/)

## What You'll Learn in This Article

- What is ONNX?
- When will you need ONNX?
- Sample ONNX projects and code
- How to quickly use models with the ONNX model zoo

## What is ONNX?

As is customary, before explaining what [ONNX](https://onnx.ai/) is and what it does, I could mention trivial facts like "it was developed by Microsoft and Facebook, supported by many developers under the Linux Foundation" that you'll never actually use. But I'd rather skip the fluff and get straight to explaining what this thing actually does.

[ONNX](https://onnx.ai/) allows us to create models using all the deep learning frameworks we know and love like [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/), and package them in a format that's supported across various hardware and operating systems. Think of it as a cross-platform API of sorts. Whether it's cloud, mobile, IoT, or wherever your mind wanders, it aims to solve problems across all these platforms.

[ONNX, which stands for "Open Neural Network Exchange"](https://onnx.ai/) is a standard developed for deep learning and various machine learning models. This standard makes AI models developed by different deep learning frameworks (such as [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), [Caffe2](https://caffe2.ai/), etc.) portable and interoperable with each other.

So, by converting a deep learning model to [ONNX](https://onnx.ai/) format, you can seamlessly use, share, and run this model across different frameworks. This gives us flexibility and makes it easier to share data and models between projects working with different frameworks (say, from PyTorch to TensorFlow).

But that's not all it brings to the table. It also tries to make the model you want to run device-agnostic (independent of CPU, GPU, FPGA, or TPU). Along with this, you're essentially leaving the optimization for different hardware to ONNX.

Additionally, the [model zoo](https://github.com/onnx/models) that I'll discuss later in the article, much like [Docker Hub](https://hub.docker.com/), allows you to quickly spin up pre-trained, ready-to-use ONNX models. What do I mean? For example, you can handle object detection with [YOLO](https://pjreddie.com/darknet/yolo/) in just a few lines and get it ready for use on any device within minutes, because someone has already prepared and packaged the entire system for you.

Before diving into the details, let me show you an example to help you visualize this.

## Possible Scenarios

Let's say you have a machine learning model that detects flower types from photos. You perform this task by uploading a photo from your computer and making predictions, using your classic x86 processor or Nvidia graphics card. However, when you want to do this on an iPhone 15 with an A16 Bionic chip, Apple will force you to use CoreML because you need to run this on the Neural Processing Unit (NPU), also known as the Apple Neural Engine.

You can quickly convert your machine learning model to a CoreML model using ONNX. You'll use the `tf2onnx` library to convert your Keras model to ONNX, and then use the `onnx-coreml` library to convert ONNX to a Core ML model. You're ready to deploy in CoreML.

### 2. TFLite on Edge Devices

You have a PyTorch model and want to convert it to TFLite to run on your edge devices. You have IoT devices or smartphones that support TFLite. You want to reduce your model size or have other similar challenges.

### 3. Other Scenarios

- You need to use AI models in your web application
- You have a project that needs to run in a Node.js environment using the `onnxruntime-node` npm package
- You want to run these models on AzureML or other cloud providers
- You want to quickly pull ready-made models from the model zoo and get them ready for use (I'll elaborate on this)

## Sample Projects

ONNX essentially highlights two main concepts. First, saving your conventional machine learning models (e.g., Scikit-Learn) in ONNX format and then using them on any device or platform. Second, converting a TensorFlow or PyTorch model to ONNX format. Let's jump into our examples without wasting time.

### Example 1: scikit-learn & ONNX

I'm bringing a basic [documentation example](https://onnx.ai/sklearn-onnx/) here and let's examine it piece by piece.

Let's make the necessary imports to build a simple [Scikit-Learn](https://scikit-learn.org/stable/index.html) model using the [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) (the national dataset of machine learning experts) and the [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) (the indispensable model of basic machine learning examples).

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
```

Let's instantiate the model and quickly fit it. If you're already here to learn ONNX, I assume you won't find it wrong that I expect you to understand these basic parts quickly.

```python
iris = load_iris()

X, y = iris.data, iris.target
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

clr = RandomForestClassifier()
clr.fit(X_train, y_train)
```

This code converts a machine learning model trained with Scikit-Learn to ONNX format and saves this converted model to a file called "rf_iris.onnx". The ONNX format will enable the model to be usable across different platforms and devices.

```python
from skl2onnx import to_onnx

onx = to_onnx(clr, X[:1])
with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

Let's load a machine learning model in ONNX format, use this model to make predictions on data, and obtain results.

```python
import onnxruntime as rt

sess = rt.InferenceSession("rf_iris.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
```

### Example 2: TensorFlow & ONNX

In this example, you can see how to convert a [TensorFlow](https://www.tensorflow.org/) or [Keras](https://keras.io/) model to ONNX format.

Now, if you're wondering why we should convert to ONNX when the TensorFlow model already works, and why we'd need this, as I mentioned above, we do this because we want to make it framework or device agnostic. Also, as I'll mention in the next blog post, you'll use this when working with platforms like [Azure](https://azure.microsoft.com/) or AWS, or if you want to build a mobile application.

```python
import tensorflow as tf
import tf2onnx

tf_model = tf.keras.models.load_model('tf_model.h5')
onnx_model = tf2onnx.convert.from_keras(tf_model)

onnx_model_path = 'converted_model.onnx'
with open(onnx_model_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("Model converted to ONNX and saved as:", onnx_model_path)
```

## More Technical Details

I mentioned that ONNX models optimize the models we're already using. ONNX runtime offers you some performance advantages.

Some of these techniques include JIT (Just-In-Time Compilation), kernel fusion, and subgraph partitioning.

Additionally, there's thread pooling support for distributed systems. These are features that can be useful when making larger-scale deployments. There's really no simple way to explain these - they're actually topics for entirely different blogs, but I'm leaving relevant links for those who are curious.

- Just-In-Time Compilation: https://www.freecodecamp.org/news/just-in-time-compilation-explained/
- Kernel Fusion: https://stackoverflow.com/questions/53305830/cuda-how-does-kernel-fusion-improve-performance-on-memory-bound-applications-on
- Subgraph Partitioning: https://en.wikipedia.org/wiki/Graph_partition
- Thread pooling: https://superfastpython.com/threadpool-python/

## ONNX Model Zoo

[ONNX Model Zoo](https://github.com/onnx/models) is a platform where volunteer developers collect various pre-trained models in ONNX format. Alongside these models, they also add Jupyter notebook examples for model training and inference with the trained models. References to the datasets used in training and original papers describing the model architecture are also included. This way, we can quickly grab and use these models without reinventing the wheel. Git LFS (Large File Storage) is used to store these models.

```python
import onnx
from onnx_tf.backend import prepare
import numpy as np
from PIL import Image
import json

# Load the ONNX model

# Use local model
model_path = './vgg19-7.onnx'
model = onnx.load(model_path)

## Download from hub
# model_path = 'vgg 19'
# model = onnx.hub.load(model_path)

# Prepare the ONNX model for TensorFlow
tf_model = prepare(model)

# Display input and output nodes of the model
print("Input nodes:", tf_model.inputs)
print("Output nodes:", tf_model.outputs)

# Load class labels from a JSON file
with open('labels.json', 'r') as json_file:
    label_data = json.load(json_file)

# Load and preprocess the input image
input_image_path = 'frog.jpg'
input_image = Image.open(input_image_path).resize((224, 224))
input_image = np.asarray(input_image, dtype=np.float32)[np.newaxis, np.newaxis, :, :]
input_image = input_image.transpose(0, 1, 4, 2, 3)
input_image = input_image.reshape(1, 3, 224, 224)  # Transform to Input Tensor

# Run inference on the input image
output = tf_model.run(input_image)[0][0]

# Find the top 5 class indices
top_indices = np.argpartition(output, -5)[-5:]

# Display the top 5 class labels and their scores
for i in top_indices:
    class_label = label_data.get(str(i), "Unknown")
    class_score = output[i]
    print(f"Class: {class_label}, Score: {class_score:.4f}")
```

```yaml
# Output

Class: American chameleon, anole, Anolis carolinensis, Score: 85.1076
Class: green lizard, Lacerta viridis, Score: 92.0642
Class: bullfrog, Rana catesbeiana, Score: 103.7489
Class: tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui, Score: 157.5779
Class: tree frog, tree-frog, Score: 186.0841
```

## Conclusion

ONNX has recently started playing an extremely important role in the AI and machine learning world, establishing itself as a standard. It solves problems between different deep learning frameworks and enables models to work and be deployed seamlessly across different platforms and devices.

_Mert Cobanov_

[Twitter](https://twitter.com/mertcobanov), [LinkedIn](https://www.linkedin.com/in/mertcobanoglu/)

> You can access all the code from this article in my [GitHub repository](https://github.com/cobanov/onnx-examples).

## References

- https://learn.microsoft.com/en-us/azure/machine-learning/concept-onnx?view=azureml-api-2
- https://onnx.ai/supported-tools
- https://blog.roboflow.com/what-is-onnx/
- https://blog.roboflow.com/what-is-tensorrt/
- https://towardsdatascience.com/onnx-the-standard-for-interoperable-deep-learning-models-a47dfbdf9a09
- https://blog.paperspace.com/what-every-ml-ai-developer-should-know-about-onnx/
- https://www.linkedin.com/pulse/what-onnx-machine-learning-model-why-should-you-care-bhattiprolu/
- https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange
- https://medium.com/geekculture/onnx-in-a-nutshell-4b584cbae7f5
- https://onnx.ai/sklearn-onnx/introduction.html
- https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
- https://viso.ai/edge-ai/tensorflow-lite/
- https://towardsdatascience.com/7-lessons-ive-learnt-from-deploying-machine-learning-models-using-onnx-3e993da4028c

---

_By Mert Cobanov on October 7, 2023_
