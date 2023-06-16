"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides a scikit-learn meta-estimator for multi-label classification that aims to maximize the example-wise F1-measure.
"""
from functools import reduce
from numbers import Number
from typing import Optional

import numpy as np

from scipy.sparse import issparse, lil_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, MultiOutputMixin, clone
from sklearn.multiclass import _estimators_has
from sklearn.utils import check_array
from sklearn.utils.metaestimators import available_if
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted


def create_gfm_weight_matrix(num_labels: int) -> np.ndarray:
    """
    Creates and returns a matrix of weights as needed by the `gfm` method.

    :param num_labels:  The total number of available labels
    :return:            An `np.ndarray` of shape (num_labels, num_labels) that has been created
    """
    return np.fromfunction(lambda s, k: 1.0 / (s + k + 2), shape=(num_labels, num_labels), dtype=np.float64)


def gfm(p: np.ndarray, p_0: float, w: Optional[np.ndarray] = None, max_cardinality: Optional[int] = None) -> np.ndarray:
    """
    Applies the General F-measure Maximizer (GFM) to a given probability matrix p.

    :param p:               An `np.ndarray` of shape (num_labels, num_labels) that constitutes a probability matrix. The
                            probability p_ik at the i-th row and the j-column specifies the conditional probability of a
                            label vector with k labels, where the i-th label is relevant
    :param p_0:             The prior probability of the empty label vector that contains no relevant labels
    :param w:               An `np.ndarray` of shape (num_labels, num_labels) that stores weights as created by the
                            method `create_gfm_weight_matrix` or None, if the weight matrix should be created implicitly
    :param max_cardinality: The maximium label cardinality observed in the training data or None, if the maximum label
                            cardinality is unknown
    :return:                An `np.ndarray` of shape (num_relevant_labels) that stores the indices of all relevant
                            labels in the label vector that maximizes the F1-measure
    """
    num_labels = p.shape[0]
    best_quality = p_0
    best_indices = np.asarray([])

    if w is None:
        w = create_gfm_weight_matrix(num_labels)

    f = np.matmul(p, w)

    for k in range(num_labels if max_cardinality is None else max_cardinality):
        indices = np.argsort(f[:, k])
        indices = indices[-(k + 1):]
        quality = 2 * np.sum(f[indices, k])

        if quality > best_quality:
            best_quality = quality
            best_indices = indices

    return best_indices


class ExampleWiseF1Maximizer(BaseEstimator, ClassifierMixin, MultiOutputMixin, MetaEstimatorMixin):
    """
    A scikit-learn meta-estimator for multi-label classification that aims to maximize the example-wise F1-measure.
    
    It transforms the original multi-label problem with n labels into a series of `n * n + 1` binary classification
    problems as described in the paper "Convex Calibrated Surrogates for the Multi-Label F-Measure" (2020) by Mingyuan
    Zhan, Harish G. Ramaswamy, and Shivani Agarwal (see http://proceedings.mlr.press/v119/zhang20w/zhang20w.pdf). A
    probabilistic base estimator is then fit to each of these independent sub-problems.
    
    The probabilities predicted by the individual base estimators for unseen examples consitute a n x n probability
    matrix p, as well as an additional probability p_0. Whereas p_0 corresponds to the prior probability of the null
    vector, i.e., a label vector that does not contain any relevant labels, each probability p_ik at the i-th row and
    k-th column of p corresponds to the conditional probability of a label vector with k relevant labels, where the i-th
    label is relevant. In order to identify the label vector that maximizes the F1 measure in expectation, these
    probabilities are used as inputs to the "General F-Measure maximizer" (GFM), as proposed in the paper "An Exact
    Algorithm for F-Measure Maximization" (2011) by Krzysztof Dembczyński, Willem Waegeman, Weiwei Cheng, and Eyke
    Hüllermeier (see https://proceedings.neurips.cc/paper/2011/file/71ad16ad2c4d81f348082ff6c4b20768-Paper.pdf).
    """

    @staticmethod
    def __transform_y_ik_sparse(y, i, k, _):
        num_examples = y.shape[0]
        indptr = y.indptr
        indices = y.indices
        nnz = 0
        transformed_y = np.zeros(shape=num_examples, dtype=np.uint8)

        for row in range(num_examples):
            start = indptr[row]
            end = indptr[row + 1]
            cardinality = end - start

            if cardinality == k:
                row_indices = indices[start:end]
                pos = np.searchsorted(row_indices, i, side='left')

                if pos < cardinality and row_indices[pos] == i:
                    nnz += 1
                    transformed_y[row] = 1

        return nnz, transformed_y

    @staticmethod
    def __transform_y_0_sparse(y, _):
        num_examples = y.shape[0]
        indptr = y.indptr
        nnz = 0
        transformed_y = np.zeros(shape=num_examples, dtype=np.uint8)

        for row in range(num_examples):
            start = indptr[row]
            end = indptr[row + 1]
            cardinality = end - start

            if cardinality == 0:
                nnz += 1
                transformed_y[row] = 1

        return nnz, transformed_y

    @staticmethod
    def __transform_y_ik_dense(y, i, k, y_cardinalities):
        num_examples = y.shape[0]
        nnz = 0
        transformed_y = np.zeros(shape=num_examples, dtype=np.uint8)

        for row in range(num_examples):
            cardinality = y_cardinalities[row]

            if cardinality == k:
                if y[row, i]:
                    nnz += 1
                    transformed_y[row] = 1

        return nnz, transformed_y

    @staticmethod
    def __transform_y_0_dense(y, y_cardinalities):
        num_examples = y.shape[0]
        nnz = 0
        transformed_y = np.zeros(shape=num_examples, dtype=np.uint8)

        for row in range(num_examples):
            cardinality = y_cardinalities[row]

            if cardinality == 0:
                nnz += 1
                transformed_y[row] = 1

        return nnz, transformed_y

    @staticmethod
    def __fit_estimator(estimator, x, y, n, transform_y_ik, transform_y_0, y_cardinalities):
        num_labels = y.shape[1]
        i = (n // num_labels)

        if i < num_labels:
            k = (n % num_labels) + 1
            nnz, transformed_y = transform_y_ik(y, i, k, y_cardinalities)
        else:
            nnz, transformed_y = transform_y_0(y, y_cardinalities)

        if nnz == 0:
            return 0
        elif nnz == y.shape[0]:
            return 1
        else:
            estimator = clone(estimator)
            estimator.fit(x, transformed_y)
            return estimator

    @staticmethod
    def __query_estimator(estimator, x):
        if isinstance(estimator, Number):
            return np.full(shape=(x.shape[0], 1), fill_value=estimator)
        else:
            p = estimator.predict_proba(x)
            shape = p.shape

            if len(shape) == 1:
                p = np.expand_dims(p, axis=1)
            elif len(shape) == 2:
                if shape[1] == 2:
                    p = np.expand_dims(p[:, 1], axis=1)
                elif shape[1] != 1:
                    raise RuntimeError('Probabilities for ' + str(shape[1] + ' classes given'))
            else:
                raise RuntimeError('Array of probabilities has shape ' + str(shape))

            return p

    def __init__(self, estimator, n_jobs=None, verbose=0):
        """
        :param estimator:   An estimator implementing `fit` and `predict_proba`
        :param n_jobs:      The number of jobs to use for fitting the estimators in parallel
        :param verbose:     The verbosity level. If non-zero, progress messages are printed. Below 50, the output is
                            sent to stderr. Otherwise, the output is sent to stdout. The frequency of the messages
                            increases with the verbosity level, reporting all iterations at 10
        """
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, x, y):
        """
        Fit underlying estimators.

        :param x:   (Sparse) array-like of shape (num_examples, num_features)
        :param y:   (Sparse) array-like of shape (num_examples, num_labels)
        :return:    Instance of fitted estimator
        """
        x = self._validate_data(x, accept_sparse=True)
        y = check_array(y, accept_sparse=True, ensure_2d=True, order='C')

        if issparse(y):
            y = y.tocsr()
            y.sort_indices()
            transform_y_ik = self.__transform_y_ik_sparse
            transform_y_0 = self.__transform_y_0_sparse
            y_cardinalities = None
            indptr = y.indptr
            self.max_cardinality_ = reduce(lambda res, i: max(res, indptr[i + 1] - indptr[i]), range(y.shape[0]), 0)
        else:
            transform_y_ik = self.__transform_y_ik_dense
            transform_y_0 = self.__transform_y_0_dense
            y_cardinalities = np.count_nonzero(y, axis=1)
            self.max_cardinality_ = np.max(y_cardinalities)

        num_labels = y.shape[1]
        self.num_labels_ = num_labels
        num_estimators = num_labels * num_labels + 1
        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self.__fit_estimator)(self.estimator, x, y, n, transform_y_ik, transform_y_0, y_cardinalities)
            for n in range(num_estimators))
        return self

    @available_if(_estimators_has("predict_proba"))
    def predict_proba(self, x):
        """
        Predict marginal probabilities required by the General F-Measure Maximizer (GFM) using underlying estimators.

        :param x:   (Sparse) array-like of shape (num_examples, num_features)
        :return:    Array-like of shape (num_examples, num_labels^2 + 1)
        """
        check_is_fitted(self)
        x = self._validate_data(x, reset=False, accept_sparse=True)
        estimators = self.estimators_
        return np.hstack(
            Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(delayed(self.__query_estimator)(estimators[n], x)
                                                               for n in range(len(estimators))))

    @available_if(_estimators_has("predict_proba"))
    def predict(self, x):
        """
        Predict binary labels that maximize the example-wise F-measure using underlying estimators.

        :param x:   (Sparse) array-like of shape (num_examples, num_features)
        :return:    A `scipy.lil_matrix` of shape (num_examples, num_labels)
        """
        max_cardinality = self.max_cardinality_
        num_labels = self.num_labels_
        num_examples = x.shape[0]
        p = self.predict_proba(x)
        w = create_gfm_weight_matrix(num_labels)
        pred = lil_matrix((num_examples, num_labels), dtype=np.uint8)
        pred_rows = pred.rows
        pred_data = pred.data

        for i in range(num_examples):
            p_row = p[i, :]
            p_0 = p_row[-1]
            p_row = np.reshape(p_row[:-1], newshape=(num_labels, num_labels))
            label_vector = gfm(p=p_row, p_0=p_0, w=w, max_cardinality=max_cardinality)
            pred_rows[i].extend(label_vector)
            pred_data[i].extend(1 for _ in range(label_vector.size))

        return pred
