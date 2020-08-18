#!/bin/bash

#
# I download the datasets for this book
#

# Get the Stack Overflow posts
nohup wget https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z &

# Get the Stack Overflow users
nohup wget https://archive.org/download/stackexchange/stackoverflow.com-Users.7z &
