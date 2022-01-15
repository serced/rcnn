## Rational Model Implementation
The rational model serves as baseline model to which we compare the comprehensiveness and sufficiency interpretability metrics. The file rationale_faithfulness.py is an adaptation of rationale.py and computes the mentioned metrics as well. In addition, we added the human annotated movie review dataset as well as code to run the model on this.

To recreate our results follow these steps:
- 1. Clone this full RCNN repo (as the rationale code relies on other parts of it)
Use: 
```bash
git clone https://github.com/serced/rcnn.git
```
- 2. Go to /code/rationale: 
```bash
cd .../code/rationale
```
- 3. Create a virtual environment based on the rationale_env.yml file: 
```bash 
conda env create -f rationale_env.yml
```
- 4. Activate the virtual environmnet 
```bash 
conda activate rationale
```
- 5. Set the Python PATH corretctly 
```bash 
cd .../code
export PYTHONPATH=${PWD} 
cd .../code/rationale 
```

- 6.1. Run the experiment using the following command: 
```bash
python rationale_faithfulness.py \
--embedding ./data/review+wiki.filtered.200.txt.gz  \
--save_model ./saved_models/testrun \
--train ./data/ \
--dev ./data/ \
--test ./data/ \
--aspect 0 \
--dump outputs.json \
--sparsity 0.0003 \
--coherent 2.0 \
--max_epochs 100 \
--learning_rate 0.001
```

- 6.2. We also provide our pretrained model, which can be accessed only via the inference command:
```bash
python rationale_faithfulness.py \
--embedding ./data/review+wiki.filtered.200.txt.gz  \
--load_model ./saved_models/testrun \
--dev ./data/ \
--test ./data/ \
--aspect 0 \
--dump outputs.json \
--sparsity 0.0003 \
--coherent 2.0 \
--max_epochs 50 \
--learning_rate 0.001
```
The rationale_faithfulness, automatically selects the human annotated movie review dataset this way.


Please find below the original README which provides further information.



## Learning Rationales behind Predictions

### About
This directory contains the code and resources of the following paper:

<i>"Rationalizing Neural Predictions". Tao Lei, Regina Barzilay and Tommi Jaakkola. EMNLP 2016.  [[PDF]](https://people.csail.mit.edu/taolei/papers/emnlp16_rationale.pdf)  [[Slides]](https://people.csail.mit.edu/taolei/papers/emnlp16_rationale_slides.pdf)</i>

The method learns to provide justifications, i.e. rationales, as supporting evidence of neural networks' prediction. The following figure illustrates the rationales and the associated predictions for multi-aspect sentiment analysis on product reivew:
<p align="center">
<img width=500 src="figures/example.png">
</p>

<br>

A **Pytorch implementation** is available: https://github.com/yala/text_nn. This version uses Gumbel softmax instead of REINFORCE for training.
<br>

A **Tensorflow implementation** is available: https://github.com/RiaanZoetmulder/Master-Thesis/tree/master/rationale (by Riaan Zoetmulder)
<br>

### Overview of the Model
We optimize two modular (neural) components, generator and encoder, to produce rationales and predictions. The framework is generic -- generator and encoder can be implemented and realized in various ways such as using RNNs or CNNs. We train the model in a RL style using policy gradient (specifically REINFORCE), as illustrated below.
<p align="center">
<img height =230 src="figures/model_framework.png">    <img width=350 src="figures/learning_framework.png">
</p>

<br>

### Sub-directories
  - this root directory contains impelmentation of the rationale model used for the beer review data. ``rationale.py`` implements the independent selection version and ``rationale_dependent.py`` implements the sequential selection version. See the paper for details.
  - [example_rationales](example_rationales) contains rationales generated for the beer review data. 
  - [ubuntu](ubuntu) contains alternative implementation for the AskUbuntu data.
  - [medical](medical) contains alternative implementation for medical report classification. 

<br>

### Data
  - **Proudct reviews:** We provide subsets of reviews and pre-trained word embeddings at [here](http://people.csail.mit.edu/taolei/beer/). This should be sufficient for producing our results. Please contact the author of the dataset, [Prof. McAuley](http://cseweb.ucsd.edu/~jmcauley/) for the full set (1.5 million reviews).   
  - **AskUbuntu data:** AskUbuntu question data is available in [this repo](https://github.com/taolei87/askubuntu).
  - **Pathology data:** This data is not available due to patients' privacy. We only provide the code and example snapshot at [/medical directory](medical)
  
**Important Note:** all data is for research-purpose only.

<br>

### Code Usage

To run the code, you need Numpy and Theano (> 0.7.0.dev-8d3a67 I used) installed. Next:
  1. Clone the rcnn repo
  2. Use “export PYTHONPATH=/path/to/rcnn/code” to add the rcnn/code directory to Python library
  3. Run `python rationale.py --help` or `python rationale_dependent.py --help` to see all running options

Example run of beer review data:
```
THEANO_FLAGS='device=gpu,floatX=float32'        # use GPU and 32-bit float
python rationale.py                             # independent selection version
      --embedding /path/to/vectors              # path to load word vectors (required)
      --train reviews.aspect0.train.txt.gz      # path to training set (required)
      --dev reviews.aspect0.heldout.txt.gz      # path to development set (required)        
      --load_rationale annotations.json         # path to rationale annotation for testing (required)
      --aspect 0                                # which aspect (-1 means all aspects)
      --dump outputs.json                       # dump selected rationales and predictions
      --sparsity 0.0003 --coherent 2.0          # regularizations
```

<br>

### To-do
  - [ ] better documentation of the code
  - [ ] more example usage of the code
  - [ ] put trained models in the repo??
