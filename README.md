# CSC413 Final Project

## Introduction:

For this project, our task was to generate a script that mimics the style and humour of the popular animated TV show Spongebob Squarepants. We accomplished this task by utilizing a long-short-term-memory (LTSM) network. LSTMs are designed to capture long-term dependencies in sequential data, and can effectively generate scripts coherent text that follows the grammar and syntax of natural language.
The input to our model is a starting word and a specified length for the generated script. The starting word serves as a prompt for the model to generate the script from, and the length indicates how many words the model should produce (where punctuation marks also count as words). With these inputs, the model was able to output a generated script with the desired length and the first word being the prompt given to it.

## Model Figure:

The architecture of this model is shown visually in Figure 1 below. The forward pass of the model is computed sequential from each of its 4 layers, in the following order: 
1. The **embedding layer** takes in the sequence of words and maps it to its integer representation in the vocabulary. In this model, our embedding layer has a dimension of 100.
2. The **LSTM layer** captures the long-term dependencies in the sequential data. This is a blackbox layer because we used Pytorch’s implementation of LSTM. However, we know from prior knowledge that this layer will take in the embedded data and process it through a series of gates that control the flow and amount of information stored in the memory cells. This enables the LSTM layer to selectively remember or forget information from previous time steps and pass forward relevant information to future time steps.
3. The **dropout layer** is a regularization technique used to prevent the model from overfitting. It works by randomly dropping out some activations during training, enabling the model to learn more robust features and become less sensitive to input noise. We used a dropout value of 0.3, meaning each of the activations in this layer have a 30% chance of being dropped out.
4. The final fully connected *linear layer* uses a linear transformation to map the output of the previous layer to an output space with the size of our vocabulary. The output of this layer, in addition to the softmax activation in script generation, is used to obtain the final prediction from a distribution of probabilities.

<img width="643" alt="image" src="https://user-images.githubusercontent.com/56453971/232602341-1d71a86e-5613-427a-87aa-f066fa2c1d9e.png">

**Figure 1:** The architecture of the forward pass for the Spongebob LSTM model. The input is an 8-gram sequence, and the output is a vector representing the probability distribution over each word in the vocabulary, given the input sequence.

## Model Parameters:

<ins>Parameters</ins>:

1. **vocab_size** - the number of unique words that appear in our training and validation set. It is also the size of our input, which is the number of features in our input.
2. **output_size** - it is the same as the vocab size; however our output includes the probabilities for each word being the next one.

<ins>Hyperparameters</ins>:

3. **seq_length** - the length of the sequence for which the last word is being predicted by the model.
4. **embedding_dim** - the size of the vectors to which our inputs (words) are being mapped
5. **hidden_dim** - the number of units in the hidden state of the LSTM layer
6. **n_layers** - the number of stacked LSTMs 
7. **dropout_rate** - the probability of a neuron being dropped in the dropout layer
8. **batch_size** - the number of data points in one batch that is created by the data loader
9. **num_epochs** - the number of completed iterations over the entire data set
10. **weight_decay** - additional term which helps to prevent overfitting
11. **learning_rate** - a rate at which the weights in our model are being updated

## Model Examples:

Our model did not use a test set to analyze its performance (more on this in “Data Split”). Instead, we measure the success of our model based on how reasonable the script generation is.
One unsuccessful example from our model is the script generation from the prompt “beaches”. Here are the first few lines of a script generated from this prompt:

beaches where. 
squidward: [ gasps as he is heard ] 
mr. krabs: [ gasps ] what's that? 
spongebob: no way, patrick, we'll never get the little twerp more, spongebob. 
spongebob: oh, yeah, i'm the most amazing! 
patrick: i can't believe that award was just a little dry! 

Clearly, the starting word became obsolete after its initial use. In fact, there is no other mention of beaches or anything related to them in the remainder of the script. 

One successful example from our model is the script generation from the prompt “spongebob”. These are the first few lines of a script generated from this prompt:

spongebob looks patrick. ". 
spongebob: [ screams and runs to the krusty krab. the little clown is still wandering in the process and down ] 
squidward: [ gasps ] what do you know. i've got a krabby patty deluxe! [ laughs ] 
mr. krabs: i'm a genius. 
spongebob: i have a good time, mr. krabs! 
mr. krabs: oh, i've been taught your new dishwasher, you dunce.
[ mr. krabs laughs. ] 
spongebob: hey, squidward!  

We can see from this example that Spongebob makes an appearance at the beginning of the episode, and the model remembered that he is still there after other characters are talking.

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

The learning curve shows that both validation and training accuracy grow rapidly from around 0-2000 iterations. After around 4000 iterations the validation accuracy plateaus; however the training accuracy keeps growing. Despite the fact that it might seem that the model starts to overfit, after judging our model based on qualitative measures we realised that this amount of iterations did not allow our model to capture some significant details. Moreover, the loss function is not very noisy and has a downward slope even after the 4000 iteration point.

<img width="600" alt="Zrzut ekranu 2023-04-17 o 16 37 41" src="https://user-images.githubusercontent.com/71830457/232605202-7176d002-130e-4ef9-b5a6-365bfa141511.png">

<img width="600" alt="Zrzut ekranu 2023-04-17 o 16 42 25" src="https://user-images.githubusercontent.com/71830457/232606204-603aa563-8820-4f32-96aa-8dbdd06757c0.png">

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

The quantitative measures we used to evaluate our model were the **training accuracy, validation accuracy and loss**. 

Firstly, the **training accuracy** was used to evaluate whether our model was correctly implemented and could perform well after appropriate hyperparameter tuning. We decided to first try overfitting it to a small data set. Once we managed to obtain 99% training accuracy on a data set containing just one file, we moved on to tuning the hyperparameters for the whole data set and evaluating our model. Then we used the **validation accuracy and loss** to evaluate how our model was performing for specific hyperparameters. 

Since the main premise of our project was to create a model that would generate scripts, we decided to not include test accuracy as one of the measures. Since the problem we decided to tackle with our model is very complex and there are a lot of words in our vocabulary, we knew that it would be incredibly hard for our model to predict the next word in a sequence correctly and give a high accuracy, especially for a never seen before test set. Moreover, we wanted to measure our model's performance based on how well it generated new scripts rather than making predictions, thus we decided not to evaluate it based on the test set, but rather some qualitative measures applied to the newly generated script. 

## Quantitative Results:

When overfitting to a small data set we obtained a training accuracy of 99% which suggested that our model was correctly implemented and had a potential to work well while generating our scripts.

Our final model had a training accuracy of 38%, validation accuracy of 29% and loss of 3.260938, which is a good result considering the complexity of our problem. Additionally, looking at the shapes of our curves, we can see that the validation accuracy plateaus at around 4000 iterations while the training accuracy continues to grow. Even though the validation accuracy plateaus and the training accuracy keeps growing it may seem that our model starts to overfit around 4000 iterations. However, when training our model on a smaller number of iterations, based on qualitative results, we noticed that our model was unable to capture important information, e.g. encapsulating actions in brackets, thus we decided to train our model with more iterations. Moreover, the loss curve keeps going down after the 4000 iterations, thus we continued training our model.

<ins>Words per line:</ins>

When for one of the generated scripts we calculated the average of words per line we got a result of an average of 12 words per line compared to 16 as stated in the data summary.

## Qualitative Results:

## Justification of Results:

### Problem Complexity: 

The task of the implemented RNN model is to generate a script for a Spongebob TV episode. This involves addressing complex dialogues such as recognizing humour, references to other characters, speech patterns, actions, and more. One of the challenging aspects of the model is understanding the context and tone of each dialogue in the script. TV show scripts typically involve multiple character interactions, with each interaction being unique to each character's personality traits and complex relationships with other characters. This is significantly difficult for the model to accurately capture. Furthermore, TV show scripts often have multiple story arcs and a specific narrative flow that the model must learn to create a structured and relevant flow for the characters and their arcs. This can be particularly challenging for the model to learn, showing the complexity of the generation problem. The model needs to be able to create dependencies based on the previous episodes, such as connecting story arcs and dialogue from previous episodes into the generated episode. These dependencies add to the complexity of the generation problem.

### Data Availability: 

There is a reasonable amount of data available to train the model. As listed in the data summary, a total of 120 episodes worth of scripts were used to train the model. The quality of the data was fair, with the scripts organized well and separated by new lines. Splitting data at newlines and punctuation would exclude those characters from the data. Initially, we made this mistake and the model would result in generating one lengthy line as the script and completely excluding new line separations. It is obvious we need the model to learn when to appropriately include newlines and punctuation, therefore during pre-processing we needed to readjust and organize the data such that we include those characters. The punctuation and the new lines are required for the model to learn when certain script lines end and a different character’s line starts.

### Model Training: 

Implementing early stopping allowed the training accuracy to not be very high and balanced it with validation accuracy, making the model more generalized. The set of hyperparameters that were analyzed and modified was the learning rate (alpha), weight decay, number of epochs, batch size, sequence length, number of stacked LSTMs, embedding dimensions and hidden dimensions. We tuned our hyperparameters based on the validation accuracy and loss for each set of hyperparameters that our model trained on, selecting the final set of parameters as; Learning rate (alpha) - 0.006, Weight Decay - 0.0, Batch Size - 800, Number of Epochs - 20, Embedding Dimensions - 100, Hidden Dimensions - 128. Achieving a good accuracy allowed the model to learn better by generalizing, and effectively structure the script, in terms of punctuation and context. For instance, using a higher number of 20 epochs allows the model to learn that actions are enclosed by square brackets, as opposed to not closing brackets with a lower number of 5 epochs. The dialogue of characters would also become more coherent as well.

### Results and Analysis:

The first line of the generated script is not coherent and does not form a structure of a typical script. After the first line, the model is able to form a script with reasonable structure and dialogue. This is likely due to the choice of the keyword, the model is unable to predict whether it should start from a script line, dialogue, or action, so the model generates the likely words before moving on to a script. The generated script is reasonably generated with good structure. Each character is named and followed by a colon before their dialogue and actions are generated. Hyperparameter tuning allowed the model to effectively place actions between square brackets and dialogue outside them. It is not always perfect, but it is consistent enough to be a reasonably structured script. The punctuation is also used appropriately, within sentences to construct a reasonable dialogue. The script is also more coherent, with most characters seeming to be conversing with other characters, about relevant topics. The conversations make sense most of the time, but occasionally some characters may appear to be conversing with themselves or referring to themselves. Other than some small discrepancies, character dialogue is reasonable. The model is also able to distinguish between the personalities of certain characters. For example, SpongeBob is a cheery character so he is laughing frequently, whereas Squidward is pessimistic and miserable, which is apparent in his tone and dialogue in the generated scripts. Overall, the model does a well enough job of recognizing the differences between characters. The model also only uses characters' names that are part of the show, to associate dialogue, therefore has an understanding of which characters frequently engage in dialogue. The end of the script usually appears cut off, and incomplete which is likely due to the length inputted. The model is unable to effectively conclude the episode and seems to create an infinite script. A hypothesis for this outcome could be, the model needs a specific length to conclude an episode, but since multiple episodes range in length and some episodes don't even indicate an ending, it could be very difficult for the model to mention an ending. 

## Ethical Considerations:

<ins>Copyright laws</ins>:

A very important issue to consider when developing such AI tools is the one of copyright laws. Since our generator learns based on previously created work, there is a consideration to be made about the ownership of the scripts created by our model. From that observation we can say that the ownership should be given to the original creators of Spongebob, since their creative work was used to train the model. However, we could also say that the work should be owned by those who created the model or should not be owned by anyone. Different countries have different laws around the ownership of “creative” work generated by AI tools. For instance, the Court of Justice of the European Union has stated that copyright can only be granted based on “author’s own intellectual creation” (Guadamuz, 2017). This is usually recognized as a necessity for a human author, thus not allowing the creators of the model to make copyright claims. On the other hand, the UK’s copyright law states that “In the case of a literary, dramatic, musical or artistic work which is computer-generated, the author shall be taken to be the person by whom the arrangements necessary for the creation of the work are undertaken” (Guadamuz, 2017). This could be understood as potentially granting copyright to both the authors of the original script and the creators of the model. Canada’s Copyright Act states that copyright can only be granted for “original work” which “must not be so trivial that it could be characterized as a purely mechanical exercise” (Parsons & Cheng, 2023). Additionally, Canada currently has no laws that address potential commercial use of work generated by AI (Parsons & Cheng, 2023). Since different countries have different views and laws concerning AI generated work, it is important to acquaint ourselves with existing laws. At the same time we should also be aware of the fact that in some cases existing laws do not cover all grounds that need to be considered to ensure that our model is being used in an ethical way. 

<ins>Problematic results</ins>:

Another important issue that should be considered is the potential of generating scripts whose tone does not agree with our values and can be deemed unethical. Since we have little to no input into what the script will generate, it could create scripts that are insensitive and harmful. For instance, in one of the examples shown above squidward mentions taking his own life, which is inappropriate to include in a children’s show. For this reason, we have to ensure that we are equipped with tools that can moderate such generated scripts. It would be important to make sure that our script goes through such a moderating tool before being used further and is also moderated by a human writer. Such filtering steps will help us moderate the content we produce and make sure that the scripts we generate are not harmful. Since our model asks for the first word in the script before generating it, it is important to note that this feature could additionally be used to generate offensive responses. To prove that this was an issue, when our generator was prompted with the word ‘kill’, it generated an inappropriate script with themes of murder. Hence, we should ensure that the moderating steps are added before using any scripts further.

## Authors:

#### Isha Kerpal

* Worked on pre-processing the data, creating the training, validation, test datasets.
* Worked on creating the RNN model and training it.
* Hyperparameter tuning while training the model.
* In the write-up I worked on Data Source, Data Summary, Data Transformation, Data Split, Hyperparameter Tuning.

#### Muhammad Idrees

* Worked on the script generation function.
* In the write-up I worked on Justification of Results.

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

We used the following as references for the Ethical Considerations section. 

* Guadamuz, A. (2017, May). A Sponge, a Starfish and the Law: Creativity and IP in the Cartoon World. WIPO Magazine. https://www.wipo.int/wipo_magazine/en/2017/05/article_0003.html
* Parsons, H., & Cheng, V. (2023, February). Who Owns AI-Generated Art? A Primer on Canadian Copyright and AI Artwork. BLG Business Law. https://www.blg.com/en/insights/2023/02/who-owns-ai-generated-art-a-primer-on-canadian-copyright-and-ai-artwork

Note: Our "get_dataloader" function is referenced from "Masrur, M. A. (2021). Friends Script Generation Using RNN-LSTM NLG [Code]. Kaggle." and our dataset is downloaded from "Gaerlan, M. (2020). SpongeBob SquarePants Completed Transcripts [Data set]. Kaggle. ".
