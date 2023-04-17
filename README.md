# CSC413

## Introduction:

## Model Figure:

## Model Parameters:

## Model Examples:

## Data Source:

The source of the dataset was Kaggle. This dataset contained scripts from nearly 393 SpongeBob episodes. Each episode script was a separate text file. The dataset can be found [here](https://www.kaggle.com/datasets/mikhailgaerlan/spongebob-squarepants-completed-transcripts?resource=download) on Kaggle.

## Data Summary:

We used a total of 200 episodes. We have 120 episodes in the training set, 40 in the validation set and 40 in the test set. Our vocabulary size (or the number of unique words) in the training and validation set is 13456. We have an average of 2696 words per episode (or per text file). We have an average of 16 words in each line or dialogue. This data summary helps us visualize the data that our model deals with. Since there is an average of 16 words per line we would want our model to generate scripts with an average sentence/dialogue length of 16 words. Since our vocabulary consists of 13456 unique words it’s an indicator of the range of different words we could possibly see in our generated script. This may seem like a small number but we need to remember that these words have to stem from the words commonly seen in SpongeBob episodes. We also see that the average length of an episode or script is 2696 words so while generating a script our word limit should be around this number. These summary statistics help us in deciding certain parameter values of our model too. For example, since we want to generate a script, we need our sentences or dialogues to make sense and be coherent with each other. Therefore it helped us decide that we should stick to a sequence length of 8 in order for our model to learn efficiently and generate a good quality script. Similarly, we also looked at the punctuation that appears in our dataset and that was an indicator of the punctuation that our generated script should also contain.

## Data Transformation:

As mentioned before our source data was text files that contained the scripts from the SpongeBob episodes. Below, we’ll highlight the data transformation process that essentially converted these text files into input that could be passed to the model for training.

### Step 1: Read Data

We looped through each file in the training set. We first remove the blank line at the top in each script. In each file, we split the lines with the newline character that is at the end of each sentence in each script. This gives us a list of sentences from all the files that we read. We then loop through each of these sentences and split them into words. We also split the sentences based on any punctuation found using the Python “re” library. The punctuation includes commas, periods, exclamation marks, question marks, square brackets and so on. Note that square brackets in our scripts denote an action being performed. Each of the words in these sentences is also converted to lowercase. We also add the newline character back to the end of each sentence so that our model can learn this, as we had initially removed it when we first read the text files. Now we get a list of lists, where each sub-list is the words (including any punctuation) in that respective sentence. This entire process is implemented in the “get_clean_text” function.

### Step 2: Create Vocabulary

Now, we create a vocabulary from the output we get from the “get_clean_text” function. We create a set of unique words from all the words that we have after reading all the files. This is done in the “get_vocab” function.

### Step 3: Create mappings of the Vocabulary

Now we create a vocabulary to integer map and a corresponding integer to vocabulary map. The “get_vocab_itos” function provides us with the corresponding integer to vocabulary map. The vocabulary to integer map is implemented in the “get_vocab_stoi” by taking in as input the integer to vocabulary map.

### Step 4: Data input for the Model

When we create our vocabulary we combine the vocabulary from the training and validation sets. This is because we need to include words from the validation set when we train so that our validation accuracy is appropriately reflected. Then we use the text parsed from the training set and get the word mappings to integers using the “vocab_stoi”. Now our text has been converted to numbers essentially which can then be passed onto the model for training purposes.

## Data Split:

We decided to use only a portion of the source data as the dataset was quite huge. We decided to use a total of 200 episode scripts for our model building process. Our data was then split into the training, validation and test sets using a 60-20-20 split in the following way:

* Training - 120 files (60% of the total number of files)
* Validation - 40 files (20% of the total number of files)
* Test - 40 files (20% of the total number of files)

Even though we split our dataset into the training, validation, test sets, we don’t calculate test accuracy as this is not appropriate to our model. This is because our model is a generative model using RNNs. We create a vocabulary of unique words from the training and validation sets to train our model and further capture how it performs on the validation set. However, since the model is trained on the vocabulary from these two sets, it cannot be used on the test set. Since the test set is supposed to contain data that has never been seen before our model doesn’t have the vocabulary from the test set and therefore it would be inappropriate to use the model on the test set as it won’t be a good predictor of how our model performs in general. Moreover, since our model’s use case is to generate new scripts, again it’s not possible to use the test set to determine the model’s efficiency. Thus, we mainly use validation accuracy to judge our model's training process and we qualitatively measure the model's performance by judging the generated script.

## Training Curve:

## Hyperparameter Tuning:

Our set of hyperparameters that we analyzed and modified were the learning rate (alpha), weight decay, number of epochs, the batch size, sequence length, number of stacked LSTMs, embedding dimensions and the hidden dimensions. We tuned our hyperparameters based on the validation accuracy and loss for each set of hyperparameters that our model trained on. 

### Learning rate (alpha):

We noticed that the optimal value for the learning rate for a batch size of 700-800 was around 0.0055-0.0075. For a batch size of 300, the optimal value for the learning rate was around 0.003-0.0045. We also noticed that for either of the batch sizes, if we decreased the learning rate to a very small value,  our loss started at a very low value and training accuracy was much higher at the end, and validation accuracy was quite low. Additionally, the loss curve and the training curve were very noisy. The loss curve also did not slope down in this case. On further analysis, we realized this could happen because the changes might be too small to the weights which cause the weights of the model to generalize well to the training set resulting in a much higher training accuracy but poor validation accuracy and generalization. 

Thus, we decided to use a learning rate of 0.006 as it was one of the optimal values for this hyperparameter for a batch size of 800. This also produced the least noisy loss curve, which sloped down well and gradually without any sharp decreases or plateaus. 

### Weight decay:

We decided to range our value for this hyperparameter from 0.001 to 0.0001. This range was based on the generally used range for weight decay while training neural networks. We decided to try a higher weight decay with a higher learning rate. This produced an extremely noisy loss curve and poor training and validation accuracy. We also tried a weight decay value of 0.0001 with the same value of the higher learning rate however the results were the same.

We then tried the values 0.001 and 0.0001 for weight decay with a lower learning rate. However, the results were the same again with extremely noisy loss curves and poor training and validation accuracies. Thus, we opted out of using any value for the weight decay hyperparameter.

### Batch Size:

We experimented with a small (200-300) and big batch size (800). We noticed that each batch size had its own optimal learning rate value. However, the optimal combination of hyperparameters for both batch sizes produced the same results in terms of loss, training and validation accuracy. With a smaller batch size we had to use a lower learning rate compared to a bigger batch size.

### Number of Epochs:

We noticed that 20-25 epochs is the optimal range for our model to train on. After 25 epochs our model would start possibly overfitting and wouldn’t generalize well to new data. Moreover, we wanted to implement “early stopping”, since our model is a generative RNN model and for it to produce a decent sounding script it was imperative that it did not learn all the fine details about the training set. Thus, we decided to use 20 epochs in our training as validation accuracy became stable after a point and loss stopped decreasing drastically.

### Embedding size and Hidden Dimensions:

We tried a few different combinations of embedding size and hidden dimensions. They all seemed to produce similar results in terms of loss, training and validation accuracy. One such example is present on the code file. We decided to stick to an embedding size of 100 and a hidden dimensions size of 128.

### Sequence Length and Number of Stacked LSTMs:

We experimented with a few different values for sequence lengths. We judged the performance based on the decency of the generated script as this was a better measure. With a smaller sequence length, our script’s sentences didn’t make sense most of the time, and would break off abruptly. So, a bigger sequence length was more desirable which is justifiable as we want to generate a script and when characters speak the sentences are much longer than 4 or 6 word sequences. Therefore, it’s important to capture that in our data and model training and we decided to use a sequence length of 8. If we increased our sequence length to more than 8 we would have less data to train on as the number of sequences would be lesser. A sequence length of 8 was a good choice of hyperparameter as it generated a decent sounding script.

### Final set of Hyperparameters:

* Learning rate (alpha) - 0.006
* Weight Decay - 0.0
* Batch Size - 800
* Number of Epochs - 20
* Embedding Dimensions - 100
* Hidden Dimensions - 128

We realize that our validation accuracy becomes a constant value, or enters a plateau around the 4000-5000 iterations mark. However, in the case of our model and problem statement, it makes more sense to keep training it for slightly longer as we want it to generate good quality scripts by being able to look at more data continuously and learn from it. This is because our loss keeps going down and our training accuracy is still going up which might be better quantitative measures to analyze. We also realize that the validation accuracy cannot be higher than a certain value due to a huge vocabulary size. The huge vocabulary size makes the probability of the next predicted word to be extremely small when we use softmax in our model. Moreover, the vocabulary from the validatios set constitutes a smaller portion of our combined vocabulary that our model uses to train on which again contributes to the fact that our model will have a low validation accuracy no matter what.

Below is a table that summarizes the various combinations of hyperparameter values that we used while training the model. The code file also demonstrates a few “bad” combinations of  hyperparameter values” to train the model and the corresponding output. At each iteration of this process, we adjusted our values according to the results of the previously tried hyperparameter values.

<img width="544" alt="image" src="https://user-images.githubusercontent.com/56453971/232345208-f5a97df7-0a9b-44d7-8600-d18b690d5a1c.png">

## Quantitative Measures:

## Quantitative and Qualitative Results:

## Justification of Results:

## Ethical Considerations:

## Authors:

#### Isha Kerpal

* Worked on pre-processing the data, creating the training, validation, test datasets.
* Worked on creating the RNN model and training it.
* Hyperparameter tuning while training the model.
* In the write-up I worked on Data Source, Data Summary, Data Transformation, Data Split, Hyperparameter Tuning.

## Advanced Concept:

Our model is a Generative RNN model as it generates a  completely new script for a SpongeBob episode. Thus, this encompasses the advanced concept for this project.

## References and Citations:

We used the following as references while designing and building our model. 

* Masrur, M. A. (2021). Friends Script Generation Using RNN-LSTM NLG [Code]. Kaggle. 
https://www.kaggle.com/code/masrur007/friends-script-generation-using-rnn-lstm-nlg
* Verma, S. (2018, April 27). Generating a TV Script Using Recurrent Neural Networks [Blog post]. Medium. 
https://shiva-verma.medium.com/generating-a-tv-script-using-recurrent-neural-networks-dd0a645e97e7
* Analatriste, C. (2017). Seinfeld Lost Episode [GitHub repository]. GitHub. 
https://github.com/cptanalatriste/seinfeld-lost-episode
* Gaerlan, M. (2020). SpongeBob SquarePants Completed Transcripts [Data set]. Kaggle. 
https://www.kaggle.com/datasets/mikhailgaerlan/spongebob-squarepants-completed-transcripts?resource=download


Note: Our "get_dataloader" function is referenced from "Masrur, M. A. (2021). Friends Script Generation Using RNN-LSTM NLG [Code]. Kaggle." and our dataset is downloaded from "Gaerlan, M. (2020). SpongeBob SquarePants Completed Transcripts [Data set]. Kaggle. ".
