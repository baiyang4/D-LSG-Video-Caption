#!/bin/bash
for i in 8 10 12; do
  for j in 6 12 18 24 30 36; do
    python train.py --num_proposals=$i --num_obj=$j --epoch_num=40
  done
done