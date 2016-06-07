#!/usr/bin/env sh
# This scripts downloads the ptb data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

mkdir -p data && cd data
wget http://russellsstewart.com/s/tensorbox/googlenet.pb
mkdir -p overfeat_rezoom && cd overfeat_rezoom
wget http://russellsstewart.com/s/tensorbox/overfeat_rezoom/save.ckpt-150000
cd ..
wget http://russellsstewart.com/s/tensorbox/brainwash.tar.gz
echo "Extracting..."
tar xf brainwash.tar.gz

echo "Done."
