#!/bin/bash

if [ $# -ne 3 ]; then
    echo "usage: $0 <label> <source> <direct>"
    exit
fi

ls -dF $2/* | sed -e "s/$/,$1/g" >> $3