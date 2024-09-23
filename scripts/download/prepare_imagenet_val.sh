#!/usr/bin/env bash
# Modified from https://gist.github.com/antoinebrl/7d00d5cb6c95ef194c737392ef7e476a

imagenet_dir=$1
cd $imagenet_dir

val_tar="${2:-ILSVRC2012_img_val.tar}"

mkdir -p val

echo "Extracting validation set ..."
tar -xvf "${val_tar}" -C val

echo "Restructuring validation ..."
cd val

# Python like zip from two streams
function zip34() { while read word3 <&3; do read word4 <&4 ; echo $word3 $word4 ; done }

wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt
find . -name "*.JPEG" | sort > images.txt
zip34 3<images.txt 4<imagenet_2012_validation_synset_labels.txt | xargs -n2 -P8 bash -c 'mkdir -p $2; mv $1 $2' argv0

rm *.txt
cd ..

echo "val:" $(find val -name "*.JPEG" | wc -l) "images"
