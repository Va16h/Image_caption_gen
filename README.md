# Neural Image Caption Generator




## Overview

This repository contains the implementation of a Neural Image Caption Generator. This model takes an image as input and generates a descriptive text caption for the image using an attention-based mechanism. The model is trained on the MS-COCO dataset and utilizes the Inception V3 model for preprocessing images and an encoder-decoder architecture for training.

## Generated Caption Example

<img align="left" width="300" height="300" src="https://github.com/Va16h/Image_caption_gen/assets/72316059/43c6ab29-334a-4694-8b83-d7b006c862d3">
 <img align="center" width="600" height="300" src="https://github.com/Va16h/Image_caption_gen/assets/72316059/d3ff40ba-3372-47ea-ac4c-54d5f9eebccf">
 <p align="center">
Generated Caption : cut up flower with a computer and a cup
</p>


## Introduction

Automatically generating textual descriptions for images is a challenging task that combines computer vision and natural language processing. This project focuses on the development of an attention-based image captioning model. Attention mechanisms allow the model to dynamically focus on specific regions of the image, resulting in more accurate and context-aware captions.

## Dataset

The dataset used for training and evaluation is the Microsoft COCO (Common Objects in Context) dataset. It comprises a vast collection of images, each annotated with multiple descriptive captions. This dataset contains images of complex scenes with various objects in their natural context. For this project, we have used a subset of 30,000 captions and their corresponding images to train the model. Utilizing a larger dataset could further enhance captioning quality.

## Pre-Processing

### Image Preprocessing

1. **Resizing and Normalization**: Images are resized to 299x299 pixels and normalized using the `preprocess_input` method. This normalization process ensures that pixel values fall within the range of -1 to 1, aligning them with the format used to train Inception V3.

2. **Feature Extraction**: Features are extracted from the last convolutional layer of the Inception V3 model. This results in a feature vector of shape (64, 2048) for each image.

3. **Caching**: Extracted image features are cached and saved to disk for efficient retrieval during training.

### Caption Preprocessing

1. **Tokenization**: Captions are tokenized by splitting them into words, creating a vocabulary of unique words in the dataset (e.g., "surfing," "football").

2. **Vocabulary Limitation**: To manage memory, the vocabulary size is limited to the top 5,000 words. All other words are replaced with the "UNK" token (unknown).

3. **Word Index Mapping**: Word-to-index and index-to-word mappings are created for efficient encoding and decoding of captions.

4. **Padding**: All caption sequences are padded to have the same length as the longest one in the dataset.

## Model Architecture

The model architecture is inspired by the "Show, Attend, and Tell" paper:

1. **Feature Extraction**: Image features extracted from Inception V3 are reshaped into a (64, 2048) vector.

2. **CNN Encoder**: A single fully connected layer acts as the CNN Encoder for image feature processing.

3. **RNN Decoder**: A GRU-based RNN Decoder attends over the image features to predict the next word in the caption.

## Training Process

The training process involves several steps:

- Extract image features from cached `.npy` files and pass them through the encoder.
- Initialize the decoder's hidden state (initialized to 0) and provide the decoder input (start token).
- Calculate predictions and decoder hidden state.
- Utilize teacher forcing, where the target word is passed as the next input to the decoder.
- Calculate gradients and apply them to the optimizer for backpropagation.

### Training Parameters

- Batch Size: 64
- Buffer Size: 1000
- Embedding Dimension: 256
- Units: 512
- Number of Epochs: 20
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy

## Results
 <p align="left">
The model generates descriptive captions for various images. Here are some examples:
</p>

<img align="left" width="300" height="300" src="https://github.com/Va16h/Image_caption_gen/assets/72316059/dc2ad68f-715f-47e5-907b-5b94249ed820">
 <img align="center" width="600" height="300" src="https://github.com/Va16h/Image_caption_gen/assets/72316059/ebaeec0c-8538-41e8-a07f-ecb994bfa6fe">
 <p align="center">
Generated Caption : Asian girl and their face stand under an umbrella
</p>
<img align="left" width="300" height="300" src="https://github.com/Va16h/Image_caption_gen/assets/72316059/ca627dbf-ce2d-4db5-84f5-b2e90144b6ba">
 <img align="center" width="600" height="300" src="https://github.com/Va16h/Image_caption_gen/assets/72316059/f8aaf5c1-58b0-40a9-bed2-307a7aa40f1b">
 <p align="center">
Generated Caption : Close-up photo taken of some vegetables and pickles.
</p>
<img align="left" width="300" height="300" src="https://github.com/Va16h/Image_caption_gen/assets/72316059/df37298c-1137-499c-b550-cee0ae0e57d5">
 <img align="center" width="600" height="300" src="https://github.com/Va16h/Image_caption_gen/assets/72316059/b74b2815-2d7c-44e8-85b0-e8914f00f61e">
 <p align="center">
Generated Caption : Desk with 2 laptop computers and a cell phone.
</p>

## Observations

The model is trained for 20 epochs, with each epoch taking approximately 30-50 seconds. While the model performs well and generates meaningful captions, training for more epochs does not significantly improve performance. Increasing the dataset size from 30,000 to 50,000 captions can enhance results, although it requires more time and storage space.

## Limitations

- The model has limited provision for handling special characters.
- Generated captions may not always be well-formulated, and further improvements can be made in future work.

## References and Credits

- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
- [TensorFlow Image Captioning Tutorial](https://www.tensorflow.org/tutorials/text/image_captioning)

This project builds upon the research papers and tutorials mentioned above.

