#!/usr/bin/env bash

# Find the file containing the string "np.int"
filename=$(find . -iname "nsgaiii.py")
echo "Found file: $filename"
# replace np.int with int 
sed -i '' -e 's/np\.int/int/g' "$filename"
cat $filename | grep "int"
