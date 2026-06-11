#!/bin/sh

set -e

cd "$(dirname "$0")"

sh train_s1.sh "$@"
sh train_s2.sh "$@"
