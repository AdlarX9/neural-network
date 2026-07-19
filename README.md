# Neural Network

This program showcases a general neural network built from scratch, and exposes by default several scripts showing off incredible performances in word prediction, number recognition and shape learning.

# Examples

## Word prediction

One of the default scripts provided by this project is word prediction. A recurrent neural network learns from a text to predict the word that is the most likely to come after a given sentence. For example, you will find in this project, a word prediction that learns only from these fifteen words:
```
roi duc duchesse prince princesse bisous amour baise lit dormir repos travail etat salaire argent
```
And after a few minutes of training of the word embedding and the neural network, it easily spits you out the full sequence from the three first words:
```
roi duc duchesse | prince princesse bisous amour baise lit dormir repos travail etat salaire argent
```

## Shape learning

> ![ovale](./examples/ovale.png)  
> *Example of a model trained to reproduce the shape on the left square, we can clearly see the model's approximation on the right, which was learnt during its training*

> ![donut](./examples/donut.png)  
> *Same thing with a more complex shape*

## Number recognition

> ![2](./examples/2.png)  
> *The number 2*

> ![6](./examples/6.png)  
> *The number 6*

> ![8](./examples/8.png)  
> *The number 8*

> ![3](./examples/3.png)  
> *The number 3, but it is not well drawn so the model hesitates with a 2*
