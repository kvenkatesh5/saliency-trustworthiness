#!/bin/bash
bash ./pipeline/hparam_lrwd.sh -a "InceptionV3" -b "32" -d "0.5" -g "4,5"
bash ./pipeline/hparam_lrwd.sh -a "InceptionV3" -b "64" -d "0.5" -g "4,5"
bash ./pipeline/hparam_lrwd.sh -a "InceptionV3" -b "128" -d "0.5" -g "4,5"
