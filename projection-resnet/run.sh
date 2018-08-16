#!/bin/bash

echo "python -u trainer.py  --arch=resnet20  --save-dir=save_resnet20 2>&1 | tee -a log_resnet20"
python -u trainer.py  --arch=resnet20  --save-dir=save_resnet20 2>&1 | tee -a log_resnet20