#!/bin/bash

PROTO=$1
WEIGHT=$2
NAME=$3



python -m mmdnn.conversion._script.convertToIR -f caffe -d $NAME -n $PROTO -w $WEIGHT
python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath $NAME.pb --dstModelPath $NAME.py -w $NAME.npy
