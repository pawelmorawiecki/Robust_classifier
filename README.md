# Towards a more robust model for road signs classification

The main idea behind this project is to use examples with random colours while training. By forcing the model to rely only on shapes, we should limit the the possibility of creating adversarial examples by small perturbations in intensity of colours. The second idea is to use a strong cryptographic hash function (one-way function), which makes impossible computing gradients with respect to the input image. Consequently, to mount the adversarial examples we need to use other model and count on the transferability argument.

![robust_model.png](attachment:robust_model.png)

Please check [Robust_classifier.ipynb](Robust_classifier.ipynb) notebook to see results of our experiments and some visualizations. 

# Implementation details

- models were trained on Belgium Traffic Signs dataset (cropped images): (http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip)
- adversarial examples were generated with Carlini-Wagner L2 attack, with Nicholas Carlini's implementation  (https://github.com/carlini/nn_robust_attacks)
- for visualization of convolutional filters we use (with slight modifictations) François Chollet's code (https://github.com/fchollet/deep-learning-with-python-notebooks), chapter 5.4
