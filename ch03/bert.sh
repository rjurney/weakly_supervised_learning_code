#!/bin/bash

# Use sentencepiece to create a WordPiece vocabulary for the data 
git clone https://github.com/google/sentencepiece
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v


VOCAB_SIZE=200000

spm_train --input=./sentences.csv --model_prefix=wsl --vocab_size=${VOCAB_SIZE}

git clone https://github.com/google-research/bert
