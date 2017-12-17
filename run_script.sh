#########################################################################
# File Name: Download_data.sh
# Author: Xianfang Zeng
# Mail: 1040804872@qq.com
# Created Time: 2017年11月05日 星期日 21时58分19秒
#########################################################################
#!/bin/bash
#Program:
#  download the data of pix2pix.

#root=/home/zxf/Project/pix2pix-tensorflow
#for dataset in cityscapes maps edges2shoes edges2handbags
#do
#  echo "Downloading dataset is ${dataset}"
#  wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/${dataset}.tar.gz
#done

#for sw in 1e4 1e5 1e6
for sw in 1e5
do
  for rw in 1e-3 1e-4 1e-5
  do
    python style_train.py --style_weight $sw --regularization_weight $rw
  done
done
