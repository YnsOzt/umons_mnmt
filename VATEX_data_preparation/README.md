# VATEX_data_preparation
Repository permettant la création des datasets VATEX pour pouvoir l'intégrer dans nmtpytorch et également télécharger les vidéos en RAW via Youtube
## pré-requis
installer jieba pour le processing des fichiers en chinois
```
pip install jieba
```

# Instruction pour télécharger les vidéos en RAW via YouTube
```
python download_video.py ./JSON_file_VATEX ./output_dir
```

## Instructions pour pré-process les fichiers
Avant de commencer, il faut switch sur IC si vous souhaitez la création du dataset pour l'IC, sinon switcher sur VMT pour la Video Machine Translation

Attention à la structure des dossiers !

0) création des données brut:
    - python dataset_creator.py ./data/base_dataset/vatex_public_test_english_v1.1.json ./data/texts/raw/test
    - python dataset_creator.py ./data/base_dataset/vatex_validation_v1.0.json ./data/texts/raw/val
    - python dataset_creator.py ./data/base_dataset/vatex_training_v1.0.json ./data/texts/raw/train


1) rester dans le dossier VATEX_data_preparation
	- bash scripts/vatex-tokenize.sh
	- EN BPE:
	  subword-nmt learn-bpe -s 10000 -i ./data/texts/tok/train.lc.norm.tok.en -o ./data/texts/bpe10k/en_code

	  subword-nmt apply-bpe -m 10000 -c ./data/texts/bpe10k/en_code -i ./data/texts/tok/train.lc.norm.tok.en -o ./data/texts/bpe10k/train.lc.norm.tok.bpe.en

	  subword-nmt apply-bpe -m 10000 -c ./data/texts/bpe10k/en_code -i ./data/texts/tok/val.lc.norm.tok.en -o ./data/texts/bpe10k/val.lc.norm.tok.bpe.en

	  subword-nmt apply-bpe -m 10000 -c ./data/texts/bpe10k/en_code -i ./data/texts/tok/test.lc.norm.tok.en -o ./data/texts/bpe10k/test.lc.norm.tok.bpe.en

	- CH BPE:
	  subword-nmt learn-bpe -s 10000 -i ./data/texts/tok/train.ch -o ./data/texts/bpe10k/ch_code

	  subword-nmt apply-bpe -m 10000 -c ./data/texts/bpe10k/ch_code -i ./data/texts/tok/train.ch -o ./data/texts/bpe10k/train.bpe.ch

	  subword-nmt apply-bpe -m 10000 -c ./data/texts/bpe10k/ch_code -i ./data/texts/tok/val.ch -o ./data/texts/bpe10k/val.bpe.ch


2) aller dans nmtpytorch
    - nmtpy-build-vocab  PATH_TO_BPE_TRAIN_FILE -o OUTPUT_DIR
