# ONNX Nedir? Neden Öğrenmelisiniz?

## Derin öğrenme modellerini her yerde kullanmanın en pratik yolu

> Bu yazıdaki tüm kodlara [github repomdan](https://github.com/cobanov/onnx-examples) ulaşabilirsiniz.

## Bu Yazıyı Kimler Okumalı

ONNX'i kullanmayı öğrenmek muhakkak ki makine öğrenimi, derin öğrenme ve yapay zeka projelerine uzanan bir işiniz varsa işe yarayacaktır, günümüzde en basit uygulamalara bile sirayet eden yapay zeka projelerinde olur da işiniz düşerse nasıl kullanabileceğinizi merak ediyorsanız veya aşağıdaki listede size uygun bir durum söz konusuysa okumanızı tavsiye ederim.

- Deep learning modelleriyle alakalı kısıtlı bilginiz varsa, web veya mobil geliştiriciyseniz
- Farklı türden cihazlarda yapay zeka modellerinizi koşturmanız gerekiyorsa
- Halihazırda kullandığınız modeller için optimizasyon yapmak istiyorsanız
- Hızlıca hazır ve eğitilmiş yapay zeka modellerini kullanmanız gerekiyorsa
- Birden fazla deep learning frameworkünü aynı uygulamada kullanmak zorundaysanız
- Bir programlama dilinde model oluşturup, ardından tamamen farklı bir çalışma ortamında olmanız gerekiyorsa (örneğin, Python'da oluşturduğunuz modeli C# veya JS'de deployment yapacaksanız)
- Muhtemelen aklıma gelmeyen fakat [dökümantasyonda](https://onnxruntime.ai/) bulacağınız diğer konular

## Bu Yazıda Neler Öğreneceksiniz

- ONNX Nedir?
- Hangi durumlarda ONNX'e ihtiyacınız olacak?
- Örnek ONNX çalışmaları ve kodları
- ONNX model zoo ile hızlıca modelleri nasıl kullanabilirsiniz?

## ONNX Nedir?

Tabii ki adettendir, [ONNX](https://onnx.ai/)'in ne olduğu ve ne yaptığı konusunda bir açıklama yapmadan önce Microsoft ve Facebook tarafından geliştirilmiştir, Linux Foundation bünyesinde birçok geliştirici tarafından desteklenmektedir gibi asla kullanmayacağınız trivial bilgileri hızlca geçip bu meretin ne yaptığını açıklamak istiyorum.

[ONNX](https://onnx.ai/), [PyTorch](https://pytorch.org/) ve [TensorFlow](https://www.tensorflow.org/) gibi bildiğimiz ve sevdiğimiz tüm deep learning frameworklerini kullanarak bir model oluşturmamıza ve çeşitli donanımlarda ve işletim sistemlerince desteklenen bir formatta paketlememize yarıyor. Aslında bu platformlar arası basit bir API gibi düşünebilirsiniz. Bulut, mobil, IoT artık aklınıza her neresi geliyorsa bu platformların tamamındaki problemlerimizi çözmeyi amaçlıyor.

[ONNX, yani uzun adıyla "Open Neural Network Exchange"](https://onnx.ai/) veya Türkçe adıyla "??Açık Sinir Ağı Değişimi??" derin öğrenme ve çeşitli makine öğrenmesi modelleri için geliştirilen bir standarttır. Bu standart, farklı derin öğrenme frameworkleri (örneğin, [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), [Caffe2](https://caffe2.ai/) vb.) tarafından geliştirilen yapay zeka modellerini taşınabilir (portable) ve birbirleriyle uyumlu (interoperable) hale getirir.

Yani, bir derin öğrenme modelini [ONNX](https://onnx.ai/) formatına dönüştürerek, bu modeli farklı frameworklerde sorunsuz bir şekilde kullanabilir, paylaşabilir ve çalıştırabilirsiniz. Bu da bize aslında bir esneklik sağlıyor ve farklı frameworklerde çalışan projeler arasında (mesela pytorch'tan tensorflow'a) veri ve model paylaşımını kolaylaştırmaya yarıyor.

Tek sihirbazlık yetenekleri de bu değil tabii ki, çalıştırmayı istediğiniz modeli istediğiniz cihazdan (CPU, GPU, FPGA veya TPU) bağımsız hale getirmeye çalışır (device-agnostic). Bununla birlikte, farklı donanımlarda optimizasyonu da ONNX'e bırakmış oluyorsunuz.

Buna ek olarak, makalenin ilerleyen kısımlarında bahsedeceğim [model zoo](https://github.com/onnx/models), aynı [Docker Hub](https://hub.docker.com/) gibi hazır eğitilmiş kullanıma hazır ONNX modellerini hızlıca ayağa kaldırmanızı sağlıyor. Ne demek istiyorum? Örneğin, [YOLO](https://pjreddie.com/darknet/yolo/) ile nesne takibi yapmayı birkaç satırda halledebilir ve istediğiniz bir cihazda dakikalar içinde kullanıma hazır hale getirebilirsiniz, çünkü biri sizin için tüm sistemi hazırlayıp paketlemiş oluyor.

Detaylara geçmeden önce bir örnek gösterip kafanızda şekillendirmek istiyorum.

## Olası Senaryolar

Diyelim ki fotoğraflardan çiçek türlerini tespit eden bir makine öğrenimi modeliniz var. Bu görevi bilgisayarınızdan bir fotoğraf yükleyerek tahminlemesini sağlıyorsunuz ve bu işlemi yaparken klasik x86 işlemcinizi veya Nvidia ekran kartınızı kullanıyorsunuz. Ancak bunu, A16 bionic çipli bir iPhone 15'te yapmak istediğinizde Apple sizi CoreML kullanmaya zorlayacak, çünkü bunu Neural Processing Unit (NPU) veya bilinen adıyla Apple Neural Engine'de yapmanız gerekiyor.

Bu makine öğrenme modelinizi CoreML modeline çevirme işlemini ONNX ile hızlıca yapabilirsiniz. `tf2onnx` kütüphanesini kullanarak Keras modelinizi ONNX'e dönüştüreceksiniz ve sonrasında `onnx-coreml` kütüphanesini kullanarak ONNX'i Core ML modeline dönüştüreceksiniz. CoreML'de deploy etmeye hazırsınız.

### 2. Edge Devicelarda TFLite

Bir PyTorch modeliniz var ve bunu edge cihazlarınızda çalışacak şekilde TFLite'a çevirmek istiyorsunuz. TFLite destekleyen IoT cihazlarınız veya akıllı telefonlarınız var. Modelinizin boyutunu düşürmek istiyorsunuz veya buna benzer farklı problemleriniz var.

### 3. Diğer senaryolar

- Web uygulamanızda yapay zeka modellerini kullanmanız gerekiyor
- `onnxruntime-node` npm paketini kullanarak Node.js ortamında çalıştırmanız gereken bir projeniz var
- AzureML veya diğer bulut sağlayıcılarda bu modelleri koşturmak istiyorsunuz
- Hazır modelleri hızlıca model zoo'dan çekip kullanıma hazır hale getirmek istiyorsunuz. (Bundan daha detaylı bahsediyorum)

## Örnek Çalışmalar

ONNX'in temelde iki konsepti oldukça ön plana çıkıyor. Birincisi, konvansiyonel makine öğrenme modellerinizi (örneğin, Scikit-Learn) ONNX formatında kaydedip sonrasında herhangi bir cihazda veya platformda kullanmak. İkincisi ise bir TensorFlow veya PyTorch modelini ONNX formatına çevirmek. Vakit kaybetmeden örneklerimize geçelim.

### Örnek 1: scikitlearn & onnx

Temel bir [dokümantasyon örneğini](https://onnx.ai/sklearn-onnx/) buraya getiriyorum ve parça parça sizinle inceleyelim.

Makine öğrenimi uzmanlarının milli veri seti olan [Iris veri seti](https://en.wikipedia.org/wiki/Iris_flower_data_set) ve temel makine öğrenimi örneklerinin vazgeçilmez modeli olan [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) ile basit bir [Scikit-Learn](https://scikit-learn.org/stable/index.html) modeli kurmak için gerekli importları yapalım.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
```

Modeli instantiate edip hızlıca modele fit edelim. Eğer zaten ONNX öğrenmek için geldiyseniz, bu temel kısımlarını hızlıca anlayacağınızı öngördüğümü yanlış bulmayacağınızı düşünüyorum.

```python
iris = load_iris()

X, y = iris.data, iris.target
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

clr = RandomForestClassifier()
clr.fit(X_train, y_train)
```

Bu kod, Scikit-Learn ile eğitilmiş bir makine öğrenme modelini ONNX formatına dönüştürüp, bu dönüştürülmüş modeli "rf_iris.onnx" adlı bir dosyaya kaydediyor. ONNX formatı, modelin farklı platformlarda ve cihazlarda kullanılabilir olmasını sağlayacak.

```python
from skl2onnx import to_onnx

onx = to_onnx(clr, X[:1])
with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

ONNX formatındaki bir makine öğrenme modelini yükleyip, bu modeli kullanarak veriler üzerinde tahminleyip sonuçları elde edelim.

```python
import onnxruntime as rt

sess = rt.InferenceSession("rf_iris.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
```

### Örnek 2: TF & ONNX

Bu örnekte, [TensorFlow](https://www.tensorflow.org/) veya [Keras](https://keras.io/) modelini ONNX formatına nasıl çevirebileceğinizi görebilirsiniz.

Peki, zaten TensorFlow modeli çalışırken neden ONNX'e çevirelim, buna neden ihtiyacımız olacak diye düşünüyorsanız, yukarıda bahsettiğim gibi bunu framework veya device agnostic hale getirmek istediğimiz için yapıyoruz. Aynı zamanda bir sonraki blog yazısında bahsedeceğim, bunu [Azure](https://azure.microsoft.com/), AWS gibi platformlarda veya bir mobil uygulama yapmak istiyorsanız kullanacaksınız.

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

## Daha Teknik Detaylar

ONNX modelinin halihazırda kullandığımız modelleri optimize ettiğinden bahsetmiştim. ONNX runtime size bazı performans avantajları sunuyor.

Bu tekniklerden biri JIT (Yani Sırası Geldiğinde Derleme), çekirdek birleştirme (kernel fusion) ve alt grafik bölümlendirme (subgraph partitioning) gibi tekniklerdir.

İkincil olarak, dağıtık sistemler için iş parçacığı havuzu (thread pooling) desteği de bulunmaktadır. Bu, daha büyük ölçekli deploymentlar yaparken işimize yarayabilecek özelliklerdir. Bunları basitçe anlatmanın pek bir yolu yok, aslında zten bunlar tamamen başka blogların konusu, fakat meraklıları için ilgili bağlantıları bırakıyorum.

- Just-In-Time Compilation: https://www.freecodecamp.org/news/just-in-time-compilation-explained/
- Kernel Fusion: https://stackoverflow.com/questions/53305830/cuda-how-does-kernel-fusion-improve-performance-on-memory-bound-applications-on
- Subgraph Partitioning https://en.wikipedia.org/wiki/Graph_partition
- Thread pooling: https://superfastpython.com/threadpool-python/

## ONNX Model Zoo

[ONNX Model Zoo](https://github.com/onnx/models), gönüllü geliştiricilerin ONNX formatında önceden eğittiği çeşitli modelleri topladığı bir platform ve bu modellerin yanında, model eğitimi ve eğitilen modelle çıkarım yapılması için Jupyter notebook örneklerini de ekliyorlar. Eğitimde kullanılan veri setleri ve model mimarisini açıklayan orijinal makalelere de referanslar ekleniyor. Bu sayede, bu modelleri tekerleği yeniden icat etmeden hızlıca alıp kullanabiliriz. Bu modelleri depolamak için Git LFS (Büyük Dosya Depolama) kullanılıyor.

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

### Sonuç

ONNX, yapay zeka ve makine öğrenme dünyasında son zamanlarda aşırı önemli bir rol oynamaya başladı ve bir standart oluşturuyor. Farklı derin öğrenme frameworkleri arasındaki problemleri çözüyor ve modellerin sorunsuz bir şekilde farklı platformlar ve cihazlar üzerinde çalışmasını ve dağıtılmasını sağlıyor.

_Mert Cobanov_

[Twitter](https://twitter.com/mertcobanov), [Linkedin](https://www.linkedin.com/in/mertcobanoglu/)

> Bu yazıdaki tüm kodlara [github repomdan](https://github.com/cobanov/onnx-examples) ulaşabilirsiniz.

## Referanslar

- https://learn.microsoft.com/tr-tr/azure/machine-learning/concept-onnx?view=azureml-api-2
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
_Exported from Medium on June 9, 2025_
