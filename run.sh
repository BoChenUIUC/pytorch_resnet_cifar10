#!/bin/bash

for model in resnet56 # resnet20 resnet32 resnet44 resnet110 resnet1202
do
    echo "python -u trainer.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model"
    #python -u trainer.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model
	python -u trainer.py --prune=save_resnet56/best.th  --arch=$model  --save-dir=save_$model |& tee -a log_$model
done