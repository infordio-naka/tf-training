#!/bin/bash

if [ $# -ne 1 ]; then
    echo "usage: $0 <file_name>"
    exit
fi

tail -n +3 $1.srt | grep "[A-Za-z]" > $1.txt