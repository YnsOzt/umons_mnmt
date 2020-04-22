[original GitHub](https://github.com/lium-lst/nmtpytorch)

## Installation
1) pip install nmtpytorch
2) python setup.py develop
3) nmtpy-install-extra

## Modèles

### VMT (Video Machine Translation)

[Fichier config](https://github.com/YnsOzt/umons_mnmt/blob/master/nmtpytorch/examples/VMT.conf)

### IMT (Image Machine Translation)

[Fichier config](https://github.com/YnsOzt/umons_mnmt/blob/master/nmtpytorch/examples/IMT.conf)

### Options disponibles
* Modifier l'encodeur:
```
* encoder_type: simple | sa | sasga # default = simple
```

* Modifier le décodeur:
```
* decoder_type: simple | trgmul # default = simple
```


* Lors de l'utilisation des features nécessitant la normalisation
```
* l2_norm: True #Si bottom-up feats mettre à true
* l2_norm_dim: -1
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

