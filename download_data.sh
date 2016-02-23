#!/usr/bin/env sh
# This scripts downloads the ptb data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

mkdir -p data && cd data
wget http://russellsstewart.com/s/tensordetect/googlenet.pb
mkdir -p lstm && cd lstm
wget http://russellsstewart.com/s/tensordetect/lstm/save.ckpt-320000
wget http://russellsstewart.com/s/tensordetect/lstm/hungarian.cc
cd ..
wget http://russellsstewart.com/s/tensordetect/brainwash.tar.gz
echo "Extracting..."
tar xf brainwash.tar.gz

echo "Done."
