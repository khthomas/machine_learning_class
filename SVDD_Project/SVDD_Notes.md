# SVDD Notes

## General
SVDD finds the smallest hyper-sphere that contains all samples (except for some outliers). 
If the kernel function has the property that k(x,x)=1∀x∈Rd, SVDD and OC-SVM learn identical decision functions. Many common kernels have this property, such as RBF, Laplacian and χ2
SVDD and OC-SVM are also equivalent in the case that all samples lie on a hypersphere centered at the origin, and are linearly separable from it.

https://datascience.stackexchange.com/questions/24335/svdd-vs-once-class-svm 

### From SK Learn
Unsupervised outlier detection

Note that in One Class SVM if no kernel is given rbf is used, aka the default is a type of SVDD

Consider a data set of observations from the same distribution described by features. 
Consider now that we add one more observation to that data set. Is the new observation 
so different from the others that we can doubt it is regular? (i.e. does it come from 
the same distribution?) Or on the contrary, is it so similar to the other that we cannot
distinguish it from the original observations? This is the question addressed by the 
novelty detection tools and methods.

### From Stack Overflow
#### Question: "What is the meaning of the nu parameter in Scikit-Learn's SVM class?
#### https://stackoverflow.com/questions/11230955/what-is-the-meaning-of-the-nu-parameter-in-scikit-learns-svm-class
Critical to the One Class SVM is the parameter nu. This is a continuous variable with values between 0 and 1. Has direct interpretation.

**Interpretation of nu**
""" The parameter nu is an upper bound on the fraction of margin errors and a lower bound of the fraction of support vectors relative to the total number of training examples. For example, if you set it to 0.05 you are guaranteed to find at most 5% of your training examples being misclassified (at the cost of a small margin, though) and at least 5% of your training examples being support vectors."""

**Relationship between C and nu**
""" The relationship between C an nu is governed by the following formula:

nu = A+B/C

A and B are constants which are unfortunately not that easy to calculate. """

*I need to talk more about C*

**Conclusion**
""" The takeaway message is that C and nu SVM are equivalent regarding their classification power. The regularization in terms of nu is easier to interpret compared to C, but the nu SVM is usually harder to optimize and runtime doesn't scale as well as the C variant with number of input samples. """

*How would one choose nu?*


## Papers:

### A Revisit to Support Vector Data Description (https://www.csie.ntu.edu.tw/~cjlin/papers/svdd.pdf)
* method for outlier detection
* There are some things that may lead to improper use of SVDD
* What is SVDD?
    1. "A model which aims at finding spherically shaped boundary around a data set."
    2. Uses a kernel matrix 
    3. There are some problems that are known in practice, but do not have a solid theoretical research
* Issues in Existing Studies of SVDD
    * Convex Optimization, Strong Duality, and KKT Conditions
        * minimize convex functions.
        * In high dimensional space, may solve the Lagrange dual problem
        * Duality gap (the Lagrange form is a lesser bound than the true convex problem, aka it is not as optimal)
        * Karush-Kuhn-Tucker (KKT) optimality conditions

### Kernel Methods: http://pub.ist.ac.at/~chl/papers/lampert-fnt2009.pdf 
"An intuitive way for outlier or anomaly detection is to estimate the density of data points in the feature space. Samples in regions of high density are typical, where samples in low density are not. However, density estimation in high dimensional spaces is a notoriously difficult problem.

## Github projects 
1. https://github.com/benmack/oneClass/blob/master/notebooks/oneClassIntro.ipynb