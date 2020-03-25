[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

`nmtpytorch` allows training of various end-to-end neural architectures including
but not limited to neural machine translation, image captioning and automatic
speech recognition systems. The initial codebase was in `Theano` and was
inspired from the famous [dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial)
codebase.

`nmtpytorch` is mainly developed by the **Language and Speech Team** of **Le Mans University** but
receives valuable contributions from the [Grounded Sequence-to-sequence Transduction Team](https://github.com/srvk/jsalt-2018-grounded-s2s)
of *Frederick Jelinek Memorial Summer Workshop 2018*:

Loic Barrault, Ozan Caglayan, Amanda Duarte, Desmond Elliott, Spandana Gella, Nils Holzenberger,
Chirag Lala, Jasmine (Sun Jae) Lee, Jindřich Libovický, Pranava Madhyastha,
Florian Metze, Karl Mulligan, Alissa Ostapenko, Shruti Palaskar, Ramon Sanabria, Lucia Specia and Josiah Wang.

If you use **nmtpytorch**, you may want to cite the following [paper](https://ufal.mff.cuni.cz/pbml/109/art-caglayan-et-al.pdf):
```
@article{nmtpy2017,
  author    = {Ozan Caglayan and
               Mercedes Garc\'{i}a-Mart\'{i}nez and
               Adrien Bardet and
               Walid Aransa and
               Fethi Bougares and
               Lo\"{i}c Barrault},
  title     = {NMTPY: A Flexible Toolkit for Advanced Neural Machine Translation Systems},
  journal   = {Prague Bull. Math. Linguistics},
  volume    = {109},
  pages     = {15--28},
  year      = {2017},
  url       = {https://ufal.mff.cuni.cz/pbml/109/art-caglayan-et-al.pdf},
  doi       = {10.1515/pralin-2017-0035},
  timestamp = {Tue, 12 Sep 2017 10:01:08 +0100}
}
```

## Installation
1) pip install nmtpytorch
2) cd nmtpytorch
3) python setup.py develop
4) nmtpy-install-extra

## Modèles

### AttentiveMNMTFeatures
Modèle avec attention simple sur l'image (features préextraite) et le texte

[Exemple de fichier de configuration](https://github.com/YnsOzt/umons_mnmt/blob/master/nmtpytorch/examples/simple_attention_txt_img.conf)

### AttentiveMNMTFeaturesTRGMUL
Modèle avec attention simple sur le texte et introduction des features visuelles via [TRGMUL](https://arxiv.org/pdf/1707.04481.pdf?fbclid=IwAR2U9oS5z3SzVUdH0aLvyEQt36-cl_MaVGT3AThqOfXPaAslr8_LUC_YlmU)

[Exemple de fichier de configuration](https://github.com/YnsOzt/umons_mnmt/blob/master/nmtpytorch/examples/TRGMUL.conf)


### AttentiveMNMTFeaturesCTXMUL
Modèle avec attention simple sur le texte et introduction des features visuelles via [CTXMUL](https://arxiv.org/pdf/1707.04481.pdf?fbclid=IwAR2U9oS5z3SzVUdH0aLvyEQt36-cl_MaVGT3AThqOfXPaAslr8_LUC_YlmU)

[Exemple de fichier de configuration](https://github.com/YnsOzt/umons_mnmt/blob/master/nmtpytorch/examples/TRGMUL.conf)


### AttentiveMNMTFeaturesSA
Modèle avec self-attention sur le texte lors de l'encodage et utilisation du décodeur multimodal avec attention sur l'image + texte

[Exemple de fichier de configuration](https://github.com/YnsOzt/umons_mnmt/blob/master/nmtpytorch/examples/TXT_Self_Attention.conf)


### AttentiveMNMTFeaturesSATRGMUL
Modèle avec self-attention sur le texte lors de l'encodage et utilisation du décodeur TRGMUL + attention_flatten

[Exemple de fichier de configuration](hhttps://github.com/YnsOzt/umons_mnmt/blob/master/nmtpytorch/examples/TXT_Self_Attention_TRGMUL.conf)


### AttentiveMNMTFeaturesSASGA
Modèle implémentant l'architecture "encoder-decoder" [du papier MCAN](https://arxiv.org/pdf/1906.10770.pdf)
  pour encoder utilisation du décodeur multimodal avec attention sur l'image + texte

[Exemple de fichier de configuration](https://github.com/YnsOzt/umons_mnmt/blob/master/nmtpytorch/examples/SA_SGA.conf)

### AttentiveMNMTFeaturesSASGATRGMUL
Modèle implémentant l'architecture "encoder-decoder" [du papier MCAN](https://arxiv.org/pdf/1906.10770.pdf)
  pour encoder et le décodeur TRGMUL + attention_flatten

[Exemple de fichier de configuration](https://github.com/YnsOzt/umons_mnmt/blob/master/nmtpytorch/examples/SA_SGA_TRGMUL.conf)

### Options lors de l'utilisation du self-attention
* ff_dim: 640 # Taille du FF layer
* dropout_sa: 0.0 # dropout du self attention
* num_sa_layers: 6 # Nombre de couche encoder et decoder
* n_head: 4 # Nombre de tête du multi-head attention
