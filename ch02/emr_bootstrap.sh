#!/bin/bash
set -x -e

# Setup Python
sudo yum -y install python-devel

# Install all required modules
sudo `which pip3` install lxml frozendict ipython pandas boto3 bs4 nltk 

# Download nltk data
python3 -m nltk.downloader punkt
python3 -m nltk.downloader stopwords

# Install Mosh for long running ssh that won't die
sudo yum -y install mosh git

# Install requirements for Snorkel processing
sudo `which pip3` install beautifulsoup4 dill gensim iso8601 jupyter numpy pandas<0.26.0 pip-tools pyarrow requests s3fs scikit-learn<0.22.0 snorkel spacy textblob textdistance texttable

# Download another model
sudo python3 -m spacy download en_core_web_lg

# Set ipython as the default shell for pyspark
export PYSPARK_DRIVER_PYTHON=ipython3
echo "" >> /home/hadoop/.bash_profile
echo "# Set ipython as the default shell for pyspark" >> /home/hadoop/.bash_profile
echo "export PYSPARK_DRIVER_PYTHON=ipython3" >> /home/hadoop/.bash_profile
echo "" >> /home/hadoop/.bash_profile
