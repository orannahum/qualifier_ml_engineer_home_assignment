## 

# create env and install requirements.txt
conda create --name qualifier-env python=3.10

conda activate qualifier-env

pip install -r requirements.txt

# create tf env and install requirements-tf.txt
conda create --name tf-quailifier-env python=3.10

conda activate tf-quailifier-env

pip install -r requirements-tf.txt

