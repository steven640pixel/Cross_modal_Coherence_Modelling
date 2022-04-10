# Cross_modal_Coherence_Modelling
The code is the implementation of Non-Autoregressive Cross-Modal Coherence Modelling.
## Prerequisites
You may need a machine with GPUs and Pytorch v1.6.0 or v1.2.0 for Python 3

1. Install Pytorch with CUDA and Python 3.6
2. Install nltk, gensim and torchvision

Our implementation adopts 300d GloVe for word embedding. 
The glove.42B.300d.txt can be downloaded from https://nlp.stanford.edu/data/glove.42B.300d.zip

##Data and Preprocessing
### Dataset
The SIND dataset can be downloaded from the Visual Storytelling website.
The sampled TACoS-Ordering dataset  can be downloaded from https://anonfiles.com/3ckfl0Tfx9/TACoS-Ordering-images_zip and https://anonfiles.com/Ndhem7Tcxb/tacos_annotation_zip
### Preprocessing
All the images are resized to 256x256 by resize.py. 

```
python resize.py --image_dir [train_image_dir] --output_dir [output_train_dir]
python resize.py --image_dir [val_image_dir] --output_dir [output_val_dir]
python resize.py --image_dir [test_image_dir] --output_dir [output_test_dir]
```

The build_vocab.py script can be used to build vocabulary.  The obtained pkl file can be placed in the voc directory.

## Training & Validation & Test
Run main.py script to train and test the model. The model.py, en_decoder.py, biatten.py and util.py are scripts that construct the model. We use torchvision to call resnet152. Change the code settings according to the modal to be ordered. 
Train the NACON models using the following command and set args.mode to train.

```
python main.py
```

Test the NACON models setting args.mode to test and the trained example models can be downloaded from the below anonymous links: https://anonfiles.com/151amcT3xd/model_example_zip


