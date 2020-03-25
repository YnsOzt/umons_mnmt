#!/bin/bash
export LC_ALL=en_US.UTF_8

if [ ! -d scripts ]; then
  echo "You should run this script from the root git folder."
  exit 1
fi

# Set path to Moses clone
MOSES="scripts/moses-3a0631a/tokenizer"
export PATH="${MOSES}:$PATH"

# Raw files path
RAW=./data/texts/raw
TOK=./data/texts/tok
PAIRS=./data/texts/pairs
SUFFIX="lc.norm.tok"

mkdir -p $TOK &> /dev/null

##############################
# Preprocess files in parallel
##############################
for TYPE in "train" "val"; do
  INP="${RAW}/${TYPE}.en"
  OUT="${TOK}/${TYPE}.${SUFFIX}.en"
  if [ -f $INP ] && [ ! -f $OUT ]; then
    cat $INP | perl ./scripts/moses-3a0631a/tokenizer/lowercase.perl | perl ./scripts/moses-3a0631a/tokenizer/normalize-punctuation.perl -l $LLANG | \
        perl ./scripts/moses-3a0631a/tokenizer/tokenizer.perl -l "en" -threads 2 > $OUT &
  fi
done
wait
