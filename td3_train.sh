#!/bin/bash

python rl_train.py --config td3.yaml
for VAR in {1..50}
do
	python rl_train.py --resume_last --config td3.yaml
done
