!/bin/bash
# pip install -r requirements.txt;
python train.py config_a > train_a.out ; python fill_gpu/fill_multi_gpu.py
# python train.py config_b > train_b.out ; python fill_gpu/fill_multi_gpu.py
# python ./fill_gpu/fill_multi_gpu.py
