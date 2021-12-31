# Few-shot-learning-CodeGolf

## CodeGolf Classification with Siamese Network

Codegolf is a programming competition to solve a problem/implement an algorithm in the fewest bytes of source code. I collected top 5000 questions/topics on <https://codegolf.stackexchange.com/questions/tagged/code-golf>, and top voted answers in Python associated to each question/topic. In this project, I attempt to build a classifier to predict which question a python code answer belongs to. For example, given the following code:
```
P=n=1
exec"P*=n*n;n+=1;"*~-input()
print P%n
```
The model predicts that this code answers the question of determining if a strictly positive integer input is prime or not.

Note there are thousands of different topics, so this would be a thousand-category classification. However, each topics only have very few accepted answers (about 2-5) for Python (also for each other programming language). Thus, building a classifier in the "normal" way (*by regularization*) is almost impossible. 

A better approach is training the classifier *by contrastive*. Siamese Network is a very interesting learning architecture in few shot learning which was orignally used in image classification [1] [2]. 

Thus, the model is built based on the spirit of Siamese Network with two variations of code embeddings, one with codeBERT representations [3], the other one with feature-based representation. Interestingly, for this particular task and setup, feature-based siamese network achieved slightly higher accuracy than codeBERT model, but much simple and faster to train...

| model                 | contrastive acc (best)  |
|-----------------------|-------------------------|
| codeBERT Siamese      | 0.78                    |
| feature-based Siamese | 0.79                    |

| model                 | 1 out of N classification (N=6) |
|-----------------------|---------------------------------|
| feature-based Siamese | 0.81                             |

Detailed evaluation see source code. 

This project contains the following files:

simple_scrapy.py -- code that scrapes all the questions urls saved to `codegolf_questions.txt` and all python code answers associated to each questions saved to `python_golfs.json`

preprocessing .py -- code that preprocesses the `python_golfs.json` and generates codeBERT embeddings for input 1 (`x1.txt`), input 2(`x2.txt`), and saves all labels to `y.txt`.

codeBERT_model.py -- run codeBERT based Siamese network model

feature_based_model_plus_eva.py -- run feature based Siamese network model

\# saved files:
codegolf_questions.txt 
python_golfs.json
x1.txt
x2.txt
y.txt

requirements.txt -- env libs

[1] *Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "Siamese neural networks for one-shot image recognition." In ICML deep learning workshop, vol. 2. 2015.*

[2] *Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "Facenet: A unified embedding for face recognition and clustering." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 815-823. 2015.*

[3] *Feng, Zhangyin, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou et al. "Codebert: A pre-trained model for programming and natural languages." arXiv preprint arXiv:2002.08155 (2020).*
