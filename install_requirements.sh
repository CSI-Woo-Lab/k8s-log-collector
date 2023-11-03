#!/bin/bash
/bin/hostname
. ~/torch_env/bin/activate

BASE_DIR=`pwd`"/"`dirname $0`
echo $BASE_DIR
EXAMPLES=`echo $1 | sed -e 's/ //g'`
echo $EXAMPLES
ERRORS=""

function error() {
  ERR=$1
  ERRORS="$ERRORS\n$ERR"
  echo $ERR
}

function install_deps() {
  echo "installing requirements"
  cat $BASE_DIR/models/*/requirements.txt | \
    sort -u | \
    # testing the installed version of torch, so don't pip install it.
    grep -vE '^torch$' | \
    pip install -r /dev/stdin || \
    { error "failed to install dependencies"; exit 1; }
}

install_deps
pip install schedule
pip install nvidia-ml-py3
pip install wget
pip install fiftyone
pip install zmq
# pip install pytorch-nlp
