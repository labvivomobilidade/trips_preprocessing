#!/bin/bash

VERSION="0.0.2"
DATA=$(date +"%d-%m-%Y - %H:%M")

read -p "Description: " DESCRIPTION

MESSAGE="Version: $VERSION - $DATA - $DESCRIPTION"

git add .
git commit -m "$MESSAGE"
git push origin main

