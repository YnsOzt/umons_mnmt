[original GitHub](https://github.com/lium-lst/nmtpytorch)

## Installation
1) pip install nmtpytorch
2) python setup.py develop
3) nmtpy-install-extra

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

[Exemple de fichier de configuration](https://github.com/YnsOzt/umons_mnmt/blob/master/nmtpytorch/examples/TXT_Self_Attention_TRGMUL.conf)


### AttentiveMNMTFeaturesSASGA
Modèle implémentant l'architecture "encoder-decoder" [du papier MCAN](https://arxiv.org/pdf/1906.10770.pdf)
  pour encoder utilisation du décodeur multimodal avec attention sur l'image + texte

[Exemple de fichier de configuration](https://github.com/YnsOzt/umons_mnmt/blob/master/nmtpytorch/examples/SA_SGA.conf)

### AttentiveMNMTFeaturesSASGATRGMUL
Modèle implémentant l'architecture "encoder-decoder" [du papier MCAN](https://arxiv.org/pdf/1906.10770.pdf)
  pour encoder et le décodeur TRGMUL + attention_flatten

[Exemple de fichier de configuration](https://github.com/YnsOzt/umons_mnmt/blob/master/nmtpytorch/examples/SA_SGA_TRGMUL.conf)

### Options supplémentaires dans le fichier de configuration
* Lors de l'utilisation des features nécessitant la normalisation
```
* l2_norm: True #Si bottom-up mettre à true
* l2_norm_dim: -1
```

* configuration du self-attention
```
* ff_dim: 640 # Taille du FF layer
* dropout_sa: 0.0 # dropout du self attention
* num_sa_layers: 6 # Nombre de couche encoder et decoder
* n_head: 4 # Nombre de tête du multi-head attention
```
* configuration de l'attention_flatten
```
* flat_mlp_size: 320 #taille intermédiaire dans le mlp du attention_flatten
```
