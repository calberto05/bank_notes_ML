"""
AUTHORS: Oliver Burguete López (A01026488),
         Carlos Alfonso Alberto Salazar (A01026175) &
         Alejandro Méndez Godoy (A01783325)
"""

Results for model KNeighborsClassifier
Accuracy: 1.0
Correct:, 412
Incorrect: 0

------------------------------------------------------

Results for model SVC
Accuracy: 0.9902912621359223
Correct:, 408
Incorrect: 4

------------------------------------------------------

Results for model Perceptron
Accuracy: 0.9927184466019418
Correct:, 409
Incorrect: 3

------------------------------------------------------

Results for model Decision Tree
Accuracy: 0.9830097087378641
Correct:, 405
Incorrect: 7







Models comparison:
The four models we worked with were: KNN, SVC, Perceptron and the one we chose, Decision Tree.
First of all, we chose the decision tree because we consider it different from the others, this method is known by its sufficiently complex decision boundaries, which helps classify the data in a correct form. And we can see the functionality with its 98% of accuracy, but it results that the method ended up being the worst out of all 4, which is comprehensible knowing that the other 3 models work better with this type of datasets.
We consider, based on the results of the accuracy score, that the KNN method is the best one for this specific problem. A major difference between KNN and the other methods is that KNN does not predict functions, it just predicts based on the proximity of n points. Furthermore, based on the analysis from the correlogram graphic, we concluded that, due to the nature of the data these fits into the KNN performance. And it makes sense that the accuracy in the majority of runs is 1, because the number of neighbors used was 5, which increases the accuracy.
On the other hand, SVC and Perceptron had almost the same performance, having around 0.99% accuracy, which is an amazing performance with just 3 or 4 incorrect predictions. We conclude that perceptron works really well in this dataset because this is an algorithm for supervised learning of binary classifiers, which in this case the variable to be predicted was presented in binary. In addition, the SVC algorithm also works quite well in separating by means of certain characteristics of binary classifications, as does perceptron, which would explain their similarity in terms of results.
