## 

# create qualifier-env and install requirements.txt
#### for all notebooks except 3-custom_neural_network.ipynb 
conda create --name qualifier-env python=3.10

conda activate qualifier-env

pip install -r requirements.txt

# create tf-qualifier-env env and install requirements-tf.txt(for tensorflow)
#### just for 3-custom_neural_network.ipynb
conda create --name tf-qualifier-env python=3.10

conda activate tf-qualifier-env

pip install -r requirements-tf.txt

