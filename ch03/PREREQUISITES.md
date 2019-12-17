# Chapter 3 Prerequisites

In order to work the Github BERT Embedding examples, you will have to download, build and install the following software:

## Google SentencePiece 

```bash
git clone https://github.com/google/sentencepiece
cd sentencepiece

mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v
```



## Google BERT

git clone https://github.com/google-research/bert
cd bert

conda create -n bert -y python=3.7.5
conda install -y pip
pip install tensorflow-gpu==1.14.0

python bert/create_pretraining_data.py \
   --input_file=data/sentences.csv \
   --output_file=data/tf_examples.tfrecord \
   --vocab_file=models/wsl.vocab \
   --bert_config_file=./bert/bert_config.json \
   --do_lower_case=False \
   --max_seq_length=128 \
   --max_predictions_per_seq=20 \
   --num_train_steps=20 \
   --num_warmup_steps=10 \
   --random_seed=1337 \
   --learning_rate=2e-5

conda deactivate