#!/bin/bash

if [ $# -ne 3 ]; then
    echo "usage: $0 <label> <source> <save_file>"
    exit
fi

ls -dF `pwd`/$2* | sed -e "s/$/,$1/g" >> $3
