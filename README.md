# Example-wise F1 Maximizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/example-wise-f1-maximizer.svg)](https://badge.fury.io/py/example-wise-f1-maximizer)

This software package provides an implementation of a meta-learning algorithm for multi-label classification that aims to maximize the example-wise F1-measure. It integrates with the popular [scikit-learn](https://scikit-learn.org) machine learning framework and can also be used with frameworks for multi-label classification like [scikit-multilearn](http://scikit.ml).

The goal of [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) is the automatic assignment of sets of labels to individual data points, for example, the annotation of text documents with topics. The example-wise [F1-measure](https://en.wikipedia.org/wiki/F-score) is a particularly relevant evaluation measure for this kind of predictions, as it requires a classifier to achieve a good balance between labels predicted as relevant or irrelevant for an example, i.e., it must neither be to conservative nor to aggressive when it comes to predicting labels as relevant.

## Methodology

The algorithm implemented by this project transforms an original multi-label problem with `n` labels into a series of `n * n + 1` binary classification problems. A probabilistic base estimator is then fit to each of these independent sub-problems as described in the following [paper](http://proceedings.mlr.press/v119/zhang20w/zhang20w.pdf):

*Mingyuan Zhan, Harish G. Ramaswamy, and Shivani Agarwal. Convex Calibrated Surrogates for the Multi-Label F-Measure. In: Proceedings of the International Conference on Machine Learning (ICML), 2020.*
    
The probabilities predicted by the individual base estimators for unseen examples consitute a `n x n` probability matrix `p`, as well as an additional probability `p_0`. Whereas `p_0` corresponds to the prior probability of the null vector, i.e., a label vector that does not contain any relevant labels, each probability `p_ik` at the `i`-th row and `k`-th column of `p` corresponds to the conditional probability of a label vector with `k` relevant labels, where the `i`-th label is relevant. In order to identify the label vector that maximizes the F1-measure in expectation, these probabilities are used as inputs to the "General F-Measure maximizer" (GFM), as proposed in the following [paper](https://proceedings.neurips.cc/paper/2011/file/71ad16ad2c4d81f348082ff6c4b20768-Paper.pdf):

*Krzysztof Dembczyński, Willem Waegeman, Weiwei Cheng, and Eyke Hüllermeier. An Exact Algorithm for F-Measure Maximization. In: Advances in Neural Information Processing Systems, 2011.*

**Please note that this implementation has not been written by any of the authors shown above.**

## Documentation

### Installation

The software package is available at [PiPy](https://pypi.org/project/example-wise-f1-maximizer/) and can easily be installed via PIP using the following command:

```
pip install example-wise-f1-maximizer
```

### Usage

To use the classifier in your own Python code, you need to import the class `ExampleWiseF1Maximizer`. It can be instantiated and used as shown below:

```python
from example_wise_f1_maximizer import ExampleWiseF1Maximizer
from sklearn.linear_model import LogisticRegression

clf = ExampleWiseF1Maximizer(estimator=LogisticRegression())
x = [[  1,  2,  3],  # Two training examples with three features
     [ 11, 12, 13]]
y = [[1, 0],  # Ground truth labels of each training example
     [0, 1]]
clf.fit(x, y)
pred = clf.predict(x)
```

The fit method accepts two inputs, `x` and `y`:

* A two-dimensional feature matrix `x`, where each row corresponds to a training example and each column corresponds to a particular feature.
* A two-dimensional binary label matrix `y`, where each row corresponds to a training examples and each column corresponds to a label. If an element in the matrix is unlike zero, it indicates that respective label is relevant to an example. Elements that are equal to zero denote irrevant labels.

Both, `x` and `y`, are expected to be [numpy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html) or equivalent [array-like](https://scikit-learn.org/stable/glossary.html#term-array-like) data types. In particular, the use of [scipy sparse matrices](https://docs.scipy.org/doc/scipy/reference/sparse.html) is supported.

In the previous example, logistic regression as implemented by the class `LogisticRegression` from the scikit-learn framework is used as a base estimator. Alternatively, you can use any probabilistic estimator for binary classification that is compatible with the scikit-learn framework and implements the `predict_proba` function.

## License

This project is open source software licensed under the terms of the [MIT license](LICENSE.md). We welcome contributions to the project to enhance its functionality and make it more accessible to a broader audience.

All contributions to the project and discussions on the [issue tracker](https://github.com/mrapp-ke/ExampleWiseF1Maximizer/issues) are expected to follow the [code of conduct](CODE_OF_CONDUCT.md).
