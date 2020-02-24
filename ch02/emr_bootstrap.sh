#!/bin/bash
set -x -e

# Install all required modules
sudo pip-3.6 install lxml frozendict ipython pandas boto3 bs4 nltk 

# Download nltk data
python3 -m nltk.downloader punkt
python3 -m nltk.downloader stopwords

# Install Mosh
sudo yum -y install mosh git

# Install requirements for Snorkel processing
sudo pip-3.6 install beautifulsoup4 dill gensim iso8601 jupyter numpy>=1.16.0 pandas pip-tools pyarrow>=0.16.0 requests s3fs scikit-learn spacy textblob textdistance texttable

# Install snorkel master
sudo pip-3.6 install -e git+git://github.com/snorkel-team/snorkel@master#egg=snorkel 

# Download another model
sudo python3 -m spacy download en_core_web_lg

# Set ipython as the default shell for pyspark
export PYSPARK_DRIVER_PYTHON=ipython3
echo "" >> /home/hadoop/.bash_profile
echo "# Set ipython as the default shell for pyspark" >> /home/hadoop/.bash_profile
echo "export PYSPARK_DRIVER_PYTHON=ipython3" >> /home/hadoop/.bash_profile
echo "" >> /home/hadoop/.bash_profile
