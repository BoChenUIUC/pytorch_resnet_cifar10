#!/bin/bash

for model in resnet56 # resnet20 resnet32 resnet44 resnet110 resnet1202
do
    echo "python -u trainer.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model"
	python -u trainer.py --prune=save_resnet56/best_base.th  --arch=$model  --save-dir=save_$model #|& tee -a log_$model
	# no norm
	#python -u trainer.py --prune=save_resnet56/best_base.th  --arch=$model --evaluate
	# L1-norm
	
	# log quantization
	#python -u trainer.py --prune=save_resnet56/best_log4.th  --arch=$model --evaluate
done