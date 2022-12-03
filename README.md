# Neural Image Compressor
Image compression using Deep learning

My simple implementation is based on [Jmtomczak 's blog](https://jmtomczak.github.io/blog/8/8_neural_compression.html).

# Set up env:
- Using conda
```commandline
conda create --name <your env name>
conda activate <your env name>
pip install -r requirement.txt
```
You can set up your hyperparameters in ```./config/config_hy```
# Train:
```commandline
python3 train.py
```
# Reconstruct:
- Reconstruct some random images in [Digits dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html):
```commandline
python3 reconstruct.py --checkpoint_path <your checkpoint received after training in ./data/checkpoint>
```
- Result will be saved at ```./images/sample_reconstruct.jpg```
# Demo:
- You can try it yourself (cuz its training phase is extremely fast :smile:)

![demo_image](./images/example_reconstruct.jpg)

# Discussion:
- <b> Quantizer seems to work like a neural network </b>    
(can we replace this block by a normal neural net which convert input into a tensor with 0,1?)
- <b> ARMEntropyEncoding's contribution is not clear for me </b>    
(where is bit stream?)
- <b> Need to read more to understand the process clearly </b>   
(maybe i should update modules with some ideas to observe the difference)  
- <b> I will try more complex models soon </b>
