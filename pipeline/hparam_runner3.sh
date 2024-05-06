#!/bin/bash
bash ./pipeline/hparam_lrwd.sh -a "DenseNet169" -b "32" -d "0" -g "6,7"
bash ./pipeline/hparam_lrwd.sh -a "DenseNet169" -b "64" -d "0" -g "6,7"
bash ./pipeline/hparam_lrwd.sh -a "DenseNet169" -b "128" -d "0" -g "6,7"