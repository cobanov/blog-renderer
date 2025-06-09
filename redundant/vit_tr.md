# Vision Transformers Nedir? ViTs Nedir?

Bu blog yazısını aynı zamanda kişisel blogumdaki [bu linkten de](https://cobanov.dev/diffusion/vit/) okuyabilirsiniz.

## Introduction

2022 yılı yapay zekanın yılı oldu, yapay zeka sanatında, doğal dil işlemede, görüntü işlemede, ses teknolojilerinde inanılmaz gelişmeler yaşandı. Hem [Huggingface](https://huggingface.co/) hem [OpenAI](https://openai.com/) fırtınalar kopardı. Önceki yıllara nazaran bu yapay zeka teknolojileri hem demokratikleşti hem de son kullanıcıya daha fazla ulaşma fırsatı buldu.

Bugün sizinle bu gelişmelerden biri olan Vision Transformerlardan bahsedeceğim. Makale giderek derinleşip teknikleşiyor, bu yüzden tüm akışı basitten karmaşığa olacak şekilde sıraladım. Bu konu ne kadar ilginizi çekiyorsa o kadar ileri gidebilirsiniz.

> Bu makaleyi yazarken birçok kaynaktan, bloglardan, kendi bilgilerimden hatta ChatGPT'den dahi yararlandım. Daha derin okumalar yapmak isterseniz lütfen kaynaklar bölümüne göz atın!

### TL; DR

2022'de Vision Transformer (ViT), şu anda bilgisayar görüşünde son teknoloji olan ve bu nedenle farklı görüntü tanıma görevlerinde yaygın olarak kullanılan evrişimli sinir ağlarına (CNN'ler) rekabetçi bir alternatif olarak ortaya çıktı.

ViT modelleri, hesaplama verimliliği ve doğruluğu açısından mevcut en son teknolojiye (CNN) neredeyse 4 kat daha iyi performans gösteriyor.

Bu makale aşağıdaki konulardan bahsedeceğim:

- Vision Transformer (ViT) nedir?
- Vision Transformers vs Convolutional Neural Networks
- Attention Mekanizması
- ViT Implementasyonları
- ViT Mimarisi
- Vision Transformers'ın Kullanım ve Uygulaması

## Vision Transformers

![ViT Overview](https://cdn-images-1.medium.com/max/800/1*jQPLjibu2eq9P1RPezpZ4A.png)

Uzun yıllardır CNN algoritmaları görüntü işleme konularında neredeyse tek çözümümüzdü. [ResNet](https://arxiv.org/abs/1512.03385), [EfficientNet](https://arxiv.org/abs/1905.11946), [Inception](https://arxiv.org/abs/1512.00567) vb. gibi tüm mimariler temelde CNN mimarilerini kullanarak görüntü işleme problemlerimizi çözmede bize yardımcı oluyordu. Bugün sizinle görüntü işleme konusunda farklı bir yaklaşım olan ViT'ler yani Vision Transformerları inceleyeceğiz.

Aslında Transformer kavramı NLP alanında yürütülen teknolojiler için ortaya kondu. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) adıyla yayınlanan makale NLP problemlerinin çözümü için devrimsel çözümler getirdi, artık Transformer-based mimaralar NLP görevleri için standart bir hale geldi.

Çok da uzun bir süre geçmeden doğal dil alanında kullanılan bu mimari görüntü alanında da ufak değişikliklerle uyarlandı. Bu çalışmayı [An image is worth 16x16 words](https://arxiv.org/abs/2010.11929) olarak linkteki paperdan okuyabilirsiniz.

Aşağıda daha detaylı anlatacağım fakat süreç temel olarak bir görüntüyü 16x16 boyutlu parçalara ayırarak ve embeddinglerini çıkartmak üzerine kuruluyor. Temel bazı konuları anlatmadan bu mekanikleri açıklamak çok zor bu yüzden hız kaybetmeden konuyu daha iyi anlamak için alt başlıklara geçelim.

## ViTs vs. CNN

Bu iki mimariyi karşılaştırdığımızda açık ara ViTlerin çok daha etkileyici olduğunu görebiliyoruz.

Vision Transformerlar, training için daha az hesaplama kaynağı kullanırken aynı zamanda, evrişimli sinir ağlarından (CNN) daha iyi performans gösteriyor.

Birazdan aşağıda daha detaylı olarak anlatacağım fakat temelde CNN'ler piksel dizilerini kullanır, ViT ise görüntüleri sabit boyutlu ufak parçalara böler. Her bir parça transformer encoder ile patch, positional vs. embeddingleri çıkartılır (aşağıda anlatacağım konular bunları içeriyor).

Ayrıca ViT modelleri, hesaplama verimliliği ve doğruluğu söz konusu olduğunda CNN'lerden neredeyse dört kat daha iyi performans gösteriyorlar.

ViT'deki self-attention katmanı, bilgilerin genel olarak görüntünün tamamına yerleştirilmesini mümkün kılıyor bu demek oluyor ki yeniden birleştirmek istediğimizde veya yenilerini oluşturmak istediğimizde bu bilgi de elimizde olacak, yani modele bunları da öğretiyoruz.

![Attention Maps](https://cdn-images-1.medium.com/max/800/1*dzEYb5Db6zlowb8Dv9lxGA.jpeg)
_Raw görselleri (solda) ViT-S/16 modeliyle attention haritaları (sağda)_

> Kaynak: [When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations](https://arxiv.org/abs/2106.01548)

## Attention Mekanizması

> Bu bölüm ChatGPT ile yazıldı

Özetle, NLP (Doğal Dil İşleme) için geliştirilen attention(dikkat) mekanizmaları, yapay sinir ağı modellerinin girdiyi daha iyi işleyip anlamasına yardımcı olmak için kullanılır. Bu mekanizmalar, girdinin farklı bölümlerine farklı ağırlık verilerek çalışır, bu sayede model girdiyi işlerken belli bölümlere daha fazla dikkat eder.

Attention mekanizmaları, dot-product attention, multi-head attention ve transformer attention gibi farklı türleri geliştirilmiştir. Bu mekanizmalar her birisi biraz farklı şekilde çalışsa da, hepsi girdinin farklı bölümlerine farklı ağırlık verilerek modelin belli bölümlere daha fazla dikkat etmesine izin verme ilkesi üzerine çalışır.

Örneğin, bir makine çeviri görevinde, bir attention mekanizması modelin kaynak dil cümlesindeki belli kelimeleri üreterek hedef dil cümlesine dikkat etmesine izin verebilir. Bu, modelin daha doğru çeviriler üretebilmesine yardımcı olur, çünkü kaynak dil kelimelerinin anlam ve bağlamını dikkate alarak çeviri üretebilir.

Genel olarak, attention mekanizmaları birçok state-of-the-art NLP modelinin bir parçasıdır ve bu modellerin çeşitli görevlerde performansını geliştirme konusunda çok etkili olduğu gösterilmiştir.

> ChatGPT sonu

Bu blogda ViT'e daha çok odaklandığımız için bu kısmı biraz hızlı geçiyorum, bu alanda hiçbir bilgisi olmayan birisi için basitleştirilmiş bir açıklama burada harika bir şekilde anlatılmış.

[Mechanics of Seq2seq Models With Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

[Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

## ViT Implementasyonları

Fine-tune edilmiş ve pre-trained ViT modelleri [Google Research](https://github.com/google-research/)'un Github'ında mevcut:

- [https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)

Pytorch Implementasyonları lucidrains'in Github reposunda bulabilirsiniz:

- [https://github.com/lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)

Aynı zamanda **timm** kullanarak hazır modelleri hızlıca kullanabilirsiniz.

- [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

## Mimari

ViT mimarisi birkaç aşamadan oluşuyor:

1. **Patch + Position Embedding (inputs):**

Giriş görüntüsünü bir dizi görüntü parçalarına (patches) dönüştürür ve parçaların hangi sırayla geldiğini bilmek için bir konum numarası ekler.

2. **Linear projection of flattened patches (Embedded Patches)**

Görüntü parçaları embeddinglere dönüştürülür, görüntüleri direkt kullanmak yerine embeddingleri kullanmanın yararı, embeddingler görüntünün eğitimle öğrenilebilir bir temsili olmasıdır.

3. **Norm**

Bir sinir ağını düzenli hale getirmek (overfitting'i azaltmak) için bir teknik olan "**Layer Normalization**" veya "**LayerNorm**"un kısaltmasıdır.

4. **Multi-Head Attention**

Bu, Multi-Headed Self-Attention layer veya kısaca "MSA" dır.

5. **MLP (Multilayer perceptron)**

Genellikle herhangi bir feed-forward (ileri besleme) katmanı koleksiyonu olarak düşünebilirsiniz.

6. **Transformer Encoder**

Transformer Encoder, yukarıda listelenen katmanların bir koleksiyonudur. Transformer Encoderin içinde iki skip (atlama) bağlantısı vardır ("+" sembolleri), katmanın girdilerinin doğrudan sonraki katmanların yanı sıra hemen sonraki katmanlara beslendiği anlamına gelir. Genel ViT mimarisi, birbiri üzerine yığılmış bir dizi Transformer kodlayıcıdan oluşur.

7. **MLP Head**

Bu, mimarinin çıktı katmanıdır, bir girdinin öğrenilen özelliklerini bir sınıf çıktısına dönüştürür. Görüntü sınıflandırması üzerinde çalıştığımız için buna "sınıflandırıcı kafa" da diyebilirsiniz. MLP head yapısı MLP bloğuna benzer.

## ViT Mimarisi

![ViT Architecture](https://cdn-images-1.medium.com/max/800/1*sL1ZF3Rt30NsK7bMArOduA.png)

### Patch Embeddings

Standart Transformer, girişi tek boyutlu token embedding dizisi olarak alır. 2B görüntüleri işlemek için **x∈R^{H×W×C}** görüntüsünü düzleştirilmiş 2B patchlere (görüntü parçalarına) yeniden şekillendiriyoruz.

Burada, (H, W) orijinal görüntünün çözünürlüğüdür ve (P, P) her görüntü parçasının çözünürlüğüdür. Resim sabit boyutlu parçalara bölünmüştür, aşağıdaki resimde patch (parça) boyutu 16×16 olarak alınmıştır. Yani görüntünün boyutları 48×48 olacaktır. (Çünkü 3 kanal var)

Self-attention maliyeti quadratictir bu yüzden görüntünün her pikselini girdi olarak iletirsek, Self-attention her pikselin diğer tüm piksellerle ilgilenmesini gerektirir. Self-attention'in ikinci dereceden (quadratic) maliyeti çok fazla olacak ve gerçekçi girdi boyutuna ölçeklenmeyecek; bu nedenle, görüntü parçalara bölünür.

Yani şair burada her pikselle uğraşmak sonsuza kadar süreceği için 16x16 boyutlu görüntü bölümlerinin embeddinglerini almanın parametre sayısını düşüreceğinden bahsediyor.

```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as npp
```

```kotlin
img = Image.open('cobanov-profile.jpg')
img.thumbnail((224, 224))
array_img = np.array(img)
array_img.shape
```

```python
# Setup hyperparameters and make sure img_size and patch_size are compatible
img_size = 224
patch_size = 16
num_patches = img_size/patch_size
assert img_size % patch_size == 0, "Image size must be divisible by patch size"

print(f"Number of patches per row: {num_patches}
Number of patches per column: {num_patches}
Total patches: {num_patches*num_patches}
Patch size: {patch_size} pixels x {patch_size} pixels")
```

```python
# Create a series of subplots
fig, axs = plt.subplots(nrows=img_size // patch_size, # need int not float
                        ncols=img_size // patch_size,
                        figsize=(num_patches, num_patches),
                        sharex=True,
                        sharey=True)

# Loop through height and width of image
for i, patch_height in enumerate(range(0, img_size, patch_size)): # iterate through height
  for j, patch_width in enumerate(range(0, img_size, patch_size)): # iterate through width

    # Plot the permuted image patch (image_permuted -> (Height, Width, Color Channels))
    axs[i, j].imshow(array_img[patch_height:patch_height+patch_size, # iterate through height
    patch_width:patch_width+patch_size, # iterate through width
    :]) # get all color channels

    # Set up label information, remove the ticks for clarity and set labels to outside
    axs[i, j].set_ylabel(i+1,
    rotation="horizontal",
    horizontalalignment="right",
    verticalalignment="center")
    axs[i, j].set_xlabel(j+1)
    axs[i, j].set_xticks([])
    axs[i, j].set_yticks([])
    axs[i, j].label_outer()

# Set a super title
plt.show()
```

![Patch Visualization](https://cdn-images-1.medium.com/max/800/1*Y2Ma0JZOSBnYo-7JiUjgwg.png)

### Linear Projection of Flattened Patches

Parçaları Transformer bloğuna geçirmeden önce, makalenin yazarları yamaları önce doğrusal bir projeksiyondan geçirmeyi faydalı bulmuşlar.

Bir yamayı alıp büyük bir vektöre açarlar ve patch embeddingler (görüntü parçalarının gömmeleri? veya embedddingleri) oluşturmak için embedding matrisiyle çarparlar ve bu konumsal gömmeyle (positional embeddings) birlikte transformatöre giden şeydir.

Her görüntü parçası (patches), tüm piksel kanallarını bir yamada birleştirerek ve ardından bunu doğrusal olarak istenen giriş boyutuna yansıtarak embed edilen bir 1B patch'e düzleştirilir.

Ne demek istediğimi bu görselde çok daha iyi anlayacağınızı düşünüyorum.

![Patch Embedding Process](https://cdn-images-1.medium.com/max/800/1*Z_zVzcIcxHSYTn7ZWdyjiQ.png)

> Kaynak: [Face Transformer for Recognition](https://arxiv.org/pdf/2103.14803.pdf)

### Positional embeddings

Nasıl konuşurken dilde kelimelerin sırası kurduğunuz cümlenin anlamını tamamen değiştiriyorsa, görüntüler üzerinde de buna dikkat etmek gerekiyor. Maalesef transformerlar, patch embeddinglerin "sırasını" dikkate alan herhangi bir varsayılan mekanizmaya sahip değiller.

Bir yapboz yaptığınızı düşünün, elinizdeki parçalar (yani önceki adımlarda yaptığımız patch embeddingler) karışık bir düzende geldiğinde görüntünün tamamında ne olduğunu anlamak oldukça zordur, bu transformerlar için de geçerli. Modelin yapboz parçalarının sırasını veya konumunu çıkarmasını sağlamanın bir yoluna ihtiyacımız var.

Transformerlar, giriş elemanlarının yapısından bağımsızdır. Her yamaya öğrenilebilir positional embeddings (konum yerleştirmeleri) eklemek, modelin, görüntünün yapısı hakkında bilgi edinmesine olanak tanır.

Positional embeddingler de, bu düzeni modele aktarmamızı sağlıyor. ViT için, bu positional embeddingler, patch embeddingler ile aynı boyutluluğa sahip öğrenilmiş vektörlerdir.

Bu positional embeddingler, eğitim sırasında ve (bazen) fine tuning sırasında öğrenilir. Eğitim sırasında, bu embeddingler, özellikle aynı sütunu ve satırı paylaşan komşu konum yerleştirmelerine yüksek benzerlik gösterdikleri vektör uzaylarında birleşir.

![Positional Embeddings](https://cdn-images-1.medium.com/max/800/1*n3d9lbZ1Uern2AyQ1ZAdrQ.png)

### Transformer Encoding

- **Multi-Head Self Attention Layer(MSP):** birden fazla attention çıktısını lineer olarak beklenen boyutlara eşitlemek için kullanılır. MSP, görüntüdeki yerel ve global bağımlılıkları öğrenmeye yardımcı olur.
- **Multi-Layer Perceptrons(MLP):** Klasik sinir ağı katmanı fakat aktivasyon fonksiyonu olarak GELU [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) kullanıyoruz.
- **Layer Norm(LN):** eğitim görüntüleri arasında herhangi bir yeni bağımlılık getirmediğinden her bloktan önce uygulanır. Eğitim süresini ve genelleme performansını iyileştirmeye yardımcı olur. Burada Misra'nin harika bir videosu var. Bu da [Paper](https://arxiv.org/abs/1607.06450).

[![Layer Normalization Video](https://img.youtube.com/vi/2V3Uduw1zwQ/0.jpg)](https://www.youtube.com/watch?v=2V3Uduw1zwQ)

- **Residual connections:** gradyanların doğrusal olmayan aktivasyonlardan geçmeden doğrudan ağ üzerinden akmasına izin verdiği için her bloktan sonra uygulanır.
- Image classification için, ön eğitim zamanında bir hidden layer ve fine-tuning için tek bir linear layer ile MLP kullanılarak bir classification head uygulanır. ViT'nin üst katmanları global özellikleri öğrenirken, alt katmanlar hem global hem de yerel özellikleri öğrenir. Bu da aslında ViT'nin daha genel kalıpları öğrenmesini sağlıyor.

## Toparlama

Evet oldukça fazla terim, teori ve mimari inceledik kafalarda daha iyi oturtmak adına bu gif sürecin nasıl işlediğini güzel bir şekilde özetliyor.

![ViT Process Animation](https://cdn-images-1.medium.com/max/800/1*_c8SqxPMY_dsApyvDJ8HtA.gif)

## Usage

Eğer basitçe bir ViT kullanmak isterseniz bunun için ufacık bir rehberi de buraya ekliyorum.

Muhtemelen denk gelmişsinizdir artık yapay zeka alanında yeni bir şey çıktığında bırakın bunun implementasyonunu, kullanımını da birkaç satır koda indirmek moda oldu.

Yukarıda anlattığım her şeyi birkaç satır Python koduyla nasıl yapıldığına bakalım.

Colab linki: [https://colab.research.google.com/drive/1sPafxIo6s1BBjHbl9e0b_DYGlb2AMBC3?usp=sharing](https://colab.research.google.com/drive/1sPafxIo6s1BBjHbl9e0b_DYGlb2AMBC3?usp=sharing)

Öncelikle pretrained model instantiate _(örneklendirmek?)_ edelim.

```python
import timm

model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()
```

Görüntümüzü yükleyip ön işlemelerini tamamlayım. Ben burada twitter profil fotoğrafımı kullanacağım.

![Profile Image](https://cdn-images-1.medium.com/max/800/1*0slpCLvJk5vEsVQwfSD5BA.jpeg)

```python
# eğer kendiniz bir görsel vermek isterseniz
# aşağıdaki kodda bu kısmı comment'e alıp,
# filename kısmına local path verebilirsiniz.
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
```

```python
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

url, filename = ("https://pbs.twimg.com/profile_images/1594739904642154497/-7kZ3Sf3_400x400.jpg", "mert.jpg")
urllib.request.urlretrieve(url, filename)

img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0) # transform and add batch dimension
```

### Tahminleri alalım

```python
import torch

with torch.no_grad():
  out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)
print(probabilities.shape)

# prints: torch.Size([1000])
```

### En popüler 5 tahminin sınıflarına bakalım

```python
# Get imagenet class mappings
url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename)

with open("imagenet_classes.txt", "r") as f:
  categories = [s.strip() for s in f.readlines()]

# Print top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
  print(categories[top5_catid[i]], top5_prob[i].item())
```

```
trench coat        0.422695130109787
bulletproof vest   0.18995067477226257
suit               0.06873432546854019
sunglasses         0.02222270704805851
sunglass           0.020680639892816544
```

## Sources

- https://www.learnpytorch.io/08_pytorch_paper_replicating/#3-replicating-the-vit-paper-an-overview
- https://theaisummer.com/vision-transformer
- https://medium.com/swlh/visual-transformers-a-new-computer-vision-paradigm-aa78c2a2ccf2
- https://viso.ai/deep-learning/vision-transformer-vit/
- https://arxiv.org/abs/2106.01548

---

_By Mert Cobanov on December 18, 2022_
