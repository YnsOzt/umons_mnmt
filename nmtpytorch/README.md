[original GitHub](https://github.com/lium-lst/nmtpytorch)

## Installation
1) pip install nmtpytorch
2) python setup.py develop
3) nmtpy-install-extra

## Données
### Bottom-up features
[Lien](https://drive.google.com/open?id=1_GRCkKv-E83QhsleukbM0mUpIg144V5J) vers les bottom-up feats (10 boxes)

* 10_boxes ==> dossier contenant les bottom up feats avec 10 box pour le train / test et val
* 36_boxes ==> dossier contenant les bottom up feats avec 36 box pour le train / test et val

### VATEX
#### VMT
[Lien](https://drive.google.com/open?id=1F84VDIsVVPxGlBc3g2SnxpBGebojy_Vv) vers les données de video machine translation

* features visuelles : ./data/videos/pre_extracted
  * train_reshaped.npy ==> contient les données d'entraînement correctement reshaped pour nmtpytorch
  * val_reshaped.npy ==> contient les données de validation correctement reshaped pour nmtpytorch
  * test_reshaped.npy ==> contient les données de test correctement reshaped pour nmtpytorch
* features textuelles : ./data/texts/bpe10k
  * train.lc.norm.tok.bpe.en ==> données d'entraînement pour l'anglais
  * train.bpe.ch ==> données d'entraînement chinois
  * val.lc.norm.tok.bpe.en ==> données de validation pour l'anglais
  * val.bpe.ch ==> données de validation chinois
  * test.lc.norm.tok.bpe.en ==> données de test en anglais
* vocabulaire : ./data/texts/bpe10k:
  * train.lc.nom.tok.bpe.vocab.en ==> vocab EN
  * train.bpe.vocab.ch ==> vocab CH


#### IC
[Lien](https://drive.google.com/open?id=1tPP6SQGMku8O4MPzosPE4HWOhzvlaeK6) vers les données d'image captionning

* features visuelles : ./data/videos/pre_extracted
  * train_reshaped.npy ==> contient les données d'entraînement correctement reshaped pour nmtpytorch
  * val_reshaped.npy ==> contient les données de validation correctement reshaped pour nmtpytorch
  * test_reshaped.npy ==> contient les données de test correctement reshaped pour nmtpytorch
* features textuelles : ./data/texts/bpe10k
  * train.lc.norm.tok.bpe.en ==> données d'entraînement pour l'anglais
  * train.bpe.ch ==> données d'entraînement chinois
  * val.lc.norm.tok.bpe.en ==> données de validation pour l'anglais
  * val.bpe.ch ==> données de validation chinois
  * test.lc.norm.tok.bpe.en ==> données de test en anglais
* vocabulaire : ./data/texts/bpe10k:
  * train.lc.nom.tok.bpe.vocab.en ==> vocab EN
  * train.bpe.vocab.ch ==> vocab CH

## Modèles

### VMT (Video Machine Translation)

[Fichier config](https://github.com/YnsOzt/umons_mnmt/blob/master/nmtpytorch/examples/VMT.conf)

### IMT (Image Machine Translation)

[Fichier config](https://github.com/YnsOzt/umons_mnmt/blob/master/nmtpytorch/examples/IMT.conf)

### Options disponibles pour la traduction
* Modifier l'encodeur:
```
* encoder_type: simple | sa | sasga # default = simple
```

* Modifier le décodeur:
```
* decoder_type: simple | trgmul # default = simple
```

* Lors de l'utilisation du self attention et guided attention
```
* ff_dim: 640 # Taille du FF layer
* dropout_sa: 0.0 # dropout du self attention
* num_layers: 6 # Nombre de couche encoder et decoder
* n_head: 4 # Nombre de tête du multi-head attention
```

* configuration de l'attention_flatten (utilisé lorsque nous utilisons un décodeur TRGMUL)
```
* flat_mlp_size: 320 #taille intermédiaire dans le mlp du attention_flatten
```


* Lors de l'utilisation des features nécessitant la normalisation
```
* l2_norm: True #Si bottom-up feats mettre à true
* l2_norm_dim: -1
```

### Video Captionning

[Fichier config](https://github.com/YnsOzt/umons_mnmt/blob/master/nmtpytorch/examples/VIC.conf)
