import functools
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
import simplejson
from pyspark.sql import SparkSession
import pickle
import re

# import AdaBoostClassifier
filename = 'trained_video_classifier.sav'


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'JSON')
        self.end_headers()

    def do_GET(self):
        print("in GET request")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello, world!')

    def do_POST(self):
        # print("in POST request")
        self._set_headers()
        self.data_string = self.rfile.read(int(self.headers['Content-Length']))
        data = simplejson.loads(self.data_string)
        # print(data)
        scoring_videos_RDD = get_scoring_videos_RDD(data)
        updated_model_decision_RDD = get_model_decision_RDD(scoring_videos_RDD)
        updated_model_decision_df = updated_model_decision_RDD.toDF(['id', 'bracket_count', 'contains_bad_language',
                                                                     'is_safe', 'final_score'])
        # updated_model_decision_df.show()
        updated_videos_json = updated_model_decision_df.toJSON().collect()
        # print(*updated_videos_json, end='\n')
        # print(type(updated_videos_json[0]))
        result_str = '[' + ','.join(map(str, updated_videos_json)) + ']'
        self.wfile.write(bytes(result_str, 'utf-8'))


def get_spark_session():
    spk = SparkSession.builder \
        .master("local") \
        .appName("Video Scoring System") \
        .config('spark.ui.port', '8080') \
        .getOrCreate()
    return spk


spark = get_spark_session()
spark_context = spark.sparkContext


def get_videos_id_bracket_count_and_is_containing_bad_words_RDD(videos_collection_json, bad_words_list):
    videos_collection_RDD = spark_context.parallelize(videos_collection_json)
    # show_RDD(videos_collection_RDD)
    videos_RDD = videos_collection_RDD.map(lambda x: (x['id'], x['parsedSubtitles']))
    # show_RDD(videos_RDD)
    # get_video_bracket_count_and_are_bad_words_contained(videos_RDD.first()[1], bad_words_list)
    videos_id_bracket_count_RDD = \
        videos_RDD.map(lambda x: (x[0],
                                  get_video_bracket_count_and_are_bad_words_contained(list(x[1]), bad_words_list))) \
            .map(lambda id_and_scoring: (id_and_scoring[0], id_and_scoring[1][0], id_and_scoring[1][1], False,
                                         get_calculated_scoring(id_and_scoring[1][0], id_and_scoring[1][1])))
    videos_id_bracket_count_RDD.cache()
    '''returning objects in this format- (id, bracket count, containing bad language, isSafePrediction, 
    final_scoring) '''
    # id_and_scoring_tuple[1,2] -> prediction, id_and_scoring_tuple[1,3] - final scoring
    show_RDD(videos_id_bracket_count_RDD)
    return videos_id_bracket_count_RDD


def get_calculated_scoring(bracket_count, contains_bad_language):
    return 10 - (bracket_count % 10) if not contains_bad_language and bracket_count <= 10 \
        else 5 - (len(str(bracket_count)) % 5)


def get_video_bracket_count_and_are_bad_words_contained(subtitles_lst, bad_wording_list):
    # Using list comprehension for mapping and counting the brackets marking cursing and bad language
    bad_words_count = 0
    are_bad_words_contained = False
    if len(subtitles_lst) != 0 and subtitles_lst is not None:
        filtered_subtitles_lst = list(filter(lambda line: line is not None, subtitles_lst))
        bad_words_count = sum(len(re.findall(r'\[.__.\]', line)) for line in filtered_subtitles_lst)
        are_bad_words_contained = any(re.search(rf'\b{bad_word}\b', line) for line in filtered_subtitles_lst \
                                      for bad_word in bad_wording_list if
                                      re.search(rf'\b{bad_word}\b', line) is not None and line is not None)
        # Debugging located bad words
        # [(i, len(re.findall(rf'\b{bad_word}\b', line)), bad_word, line) for i, line in enumerate(subtitles_lst) for
        #  bad_word in bad_wording_list if len(re.findall(bad_word, line)) != 0]

        # words_lists = [line.split(" ") if line is not None else "" for line in subtitles_lst]
        # # Assuming almost no words containing '__' but the '[ __ ]' after splitting by ' ' that
        # # indicates there has been a bad word in the same place marked with this sequence of characters.
        # if len(words_lists) != 0 and words_lists is not None:
        #     # Checks whether there is a bad word for the bad word list appeared in the list of words
        #     # of the current transcript
        #     flattened_words_list = [word for words_lst in words_lists for word in words_lst]
        #     are_bad_words_contained = any(bad_word in flattened_words_list for bad_word in bad_wording_list)
        #     bad_words_count_prep = [[int(1) if word == "__" else int(0) for word in words_lst]
        #                             for words_lst in words_lists]
        #     # print(*bad_words_count_prep, end='\n')
        #     # print(str(len(bad_words_count_prep)) + '\n')
        #     # Appending bracket counting lists an the counting the number of brackets in total (summing 0's and 1's)
        #     if len(bad_words_count_prep) != 0 and bad_words_count_prep is not None:
        #         bad_words_count = functools.reduce(lambda a, b: a + b,
        #                                            functools.reduce(lambda a, b: a + b, bad_words_count_prep))
    print(f'count is {bad_words_count}\n')
    print(f'contains bad words: {are_bad_words_contained}\n')

    return bad_words_count, are_bad_words_contained


def show_RDD(rdd):
    spark.read.json(rdd).show(20, truncate=False)


def get_videos_collection_resp(get_request_route):
    response = requests.get(get_request_route)
    # Check
    print(f'first json object: {response.json()[0]}\n')
    return response


def get_videos_collection_keys(response):
    # Check
    print(f'keys: {response.json()[0].keys()}\n')
    return response.json()[0].keys()


def get_model_decision_RDD(updated_videos_RDD):
    loaded_classifier = pickle.load(open(filename, 'rb'))
    # Our model wasn't accurate enough so we mad a small change by subtracting 4 out of the bracket count to predict 1
    # when the counter was under 5 and -1 when the counter was5 or higher than that.
    # show_RDD(updated_videos_RDD.map(lambda x: x[1]))
    updated_RDD = updated_videos_RDD.map(
        lambda x: (
            x[0], x[1], x[2], True if loaded_classifier.predict(np.array((575, x[1] - 4)).reshape((1, -1))) == 1 \
                else False, x[4]))
    # show_RDD(updated_RDD)
    updated_RDD.map(lambda x: (x[0], x[1], x[2], x[3], x[4] - 5) if not x[3] and x[4] > 5 else x[4])
    show_RDD(updated_RDD)
    return updated_RDD


def get_scoring_videos_RDD(response_data=None):
    # resp = get_videos_collection_resp('http://localhost:8082/api/v1/youtube/getalltranscripts')
    # videos_keys = get_videos_collection_keys(resp)
    bad_words_df = spark.read.csv('Bad_Words_Dataset/bad-words.csv')
    bad_words_list = bad_words_df.select('_c0').rdd.map(lambda row: row[0]).collect()
    updated_videos_RDD = get_videos_id_bracket_count_and_is_containing_bad_words_RDD(response_data,
                                                                                     bad_words_list)
    # updated_videos_RDD = get_videos_id_bracket_count_and_is_containing_bad_words_RDD(resp.json(),
    #                                                                                  bad_words_list)

    # print(f'type json element after updating {type(updated_videos_json)}\n')
    return updated_videos_RDD


# -------------------------------Classifier--------------------------------------

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import fmin_tnc
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

current_option_number = 1
spark = get_spark_session()
spark_context = spark.sparkContext


def plot_adaboost(_X: np.ndarray,
                  _y: np.ndarray,
                  clf=None,
                  sample_weights: Optional[np.ndarray] = None,
                  annotate: bool = False,
                  ax: Optional[mpl.axes.Axes] = None,
                  base_clf_errors=None,
                  base_clf_stump_weights=None) -> None:
    """ Plot ± samples in 2D, optionally with decision boundary
    :param base_clf_errors: all T errors from the training process
    :param ax: axes for plotting
    :param annotate:
    :param clf: The Adaboost generated classifier
    :param _y: labels
    :param _X: samples
    :param sample_weights: weights for training process
    :param base_clf_stump_weights: coefficient of all T base classifiers
    """

    assert set(_y) == {-1, 1}, 'Expecting response labels to be ±1'

    if not ax:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        fig.set_facecolor('white')

    pad = 1
    x_min, x_max = _X[:, 0].min() - pad, _X[:, 0].max() + pad
    y_min, y_max = _X[:, 1].min() - pad, _X[:, 1].max() + pad

    if sample_weights is not None:
        sizes = np.array(sample_weights) * _X.shape[0] * 100
    else:
        sizes = np.ones(shape=_X.shape[0]) * 100

    X_pos = _X[_y == 1]
    sizes_pos = sizes[_y == 1]
    ax.plot([x1 for x1, x2 in list(X_pos)], [x2 for x1, x2 in list(X_pos)], '+r',
            label='sample group 1')

    X_neg = _X[_y == -1]
    sizes_neg = sizes[_y == -1]
    ax.plot([x2 for x1, x2 in list(X_neg)], [x2 for x1, x2 in list(X_neg)], 'ob',
            label='sample group 2')
    if base_clf_errors is not None:
        (x_min, x_max), (y_min, y_max) = plot_collection_by_t(np.asarray(list(range(len(base_clf_errors)))),
                                                              np.asarray(base_clf_errors), ax=ax, value_name="error",
                                                              min_sample_val=x_min,
                                                              max_sample_val=x_max,
                                                              min_label_val=y_min, max_label_val=y_max, marker='*',
                                                              color='y', )

    if base_clf_stump_weights is not None:
        plot_collection_by_t(np.asarray(list(range(len(base_clf_stump_weights)))),
                             np.asarray(base_clf_stump_weights), ax=ax, value_name="alpha", min_sample_val=x_min,
                             max_sample_val=x_max,
                             min_label_val=y_min, max_label_val=y_max, marker='^', color='g')

    # if clf:
    #     plot_step = 0.01
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
    #                          np.arange(y_min, y_max, plot_step))
    #     Z = np.zeros(shape=xx.shape)
    #     if isinstance(clf, AdaBoost) or isinstance(clf, GaussianNB):  # AdaBoost
    #         Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #     elif current_option_number == 2 and isinstance(clf, tuple):
    #         if isinstance(clf[0], type(LogisticRegression)):
    #             clf = clf[0]
    #             theta = clf[1]
    #             Z = clf.predict(np.c_[xx.ravel(), yy.ravel()], theta)
    #     Z = Z.reshape(xx.shape)
    #
    #     # If all predictions are positive class, adjust color map accordingly
    #     if list(np.unique(Z)) == [1]:
    #         fill_colors = ['r']
    #     else:
    #         fill_colors = ['b', 'r']
    #
    #     ax.contourf(xx, yy, Z, colors=fill_colors, alpha=0.2)

    if annotate:
        for i, (x, y_annotate) in enumerate(X):
            offset = 0.05
            ax.annotate(f'$x_{i + 1}$', (x + offset, y_annotate - offset))

    ax.set_xlim(x_min + 0.5, x_max - 0.5)
    ax.set_ylim(y_min + 0.5, y_max - 0.5)
    ax.set_xlabel('x / iteration number')
    ax.set_ylabel('$y$')


def plot_collection_by_t(x_values: np.ndarray,
                         y_values: np.ndarray,
                         ax: Optional[mpl.axes.Axes] = None,
                         value_name="y",
                         min_sample_val=0,
                         max_sample_val=0,
                         min_label_val=0,
                         max_label_val=0,
                         marker="+",
                         color="blue"):
    pad = 1

    x_min, x_max = min(x_values.min(), min_sample_val) - pad, max(x_values.max(), max_sample_val) + pad
    y_min, y_max = min(y_values.min(), min_label_val) - pad, max(y_values.max(), max_label_val) + pad
    print(type(ax))
    ax.plot(x_values, y_values, marker + color, label=value_name)
    ax.set_xlim(x_min + 0.5, x_max - 0.5)
    ax.set_ylim(y_min + 0.5, y_max - 0.5)
    ax.legend()

    return (x_min, x_max), (y_min, y_max)


def truncate_adaboost(clf, t: int):
    """ Truncate a fitted AdaBoost up to (and including) a particular iteration """
    assert t > 0, 't must be a positive integer'
    new_clf = deepcopy(clf)

    if current_option_number == 1:
        # ------------------------Decision Trees Classifiers only------------------------------------
        new_clf.stumps = clf.stumps[:t]
        new_clf.stump_weights = clf.stump_weights[:t]
        # ------------------------Decision Trees Classifiers only------------------------------------

    return new_clf


def get_mapped_transcript_dataset(scoring_videos_json):
    # Referring to 1 as suitable for kids and -1 as not suitable (labels)
    X_and_Y_RDD = spark_context.parallelize([json.loads(scored_vid) for scored_vid in scoring_videos_json]) \
        .map(lambda scored_vid: (scored_vid['bracket_count'], 1) if scored_vid['final_score'] > 4
    else (scored_vid['bracket_count'], -1)).zipWithIndex()
    X_and_Y_RDD.cache()
    # Samples are represented by (indexed video, bracket_count) and labels are 1 for suitable and -1 for not suitable
    X_dataset, y_dataset = np.array(X_and_Y_RDD.map(lambda x: (x[1], x[0][0])).collect()), \
                           np.array(X_and_Y_RDD.map(lambda x: x[0][1]).collect())
    return X_dataset, y_dataset


class AdaBoost:
    """ AdaBoost enemble classifier from scratch """

    def __init__(self):
        #  ------------------------Decision Trees Classifiers only------------------------------------
        self.stumps = None
        self.stump_weights = None
        #  ------------------------Decision Trees Classifiers only------------------------------------
        self.errors = None
        self.sample_weights = None


    @staticmethod
    def _check_X_y(X, y):
        """ Validate assumptions about format of input data"""
        assert set(y) == {-1, 1}, 'Response variable must be ±1'
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray, iters: int):
        """ Fit the model using training data """

        # X, y = self._check_X_y(X, y)
        n = X.shape[0]

        """ init numpy arrays """
        self.sample_weights = np.zeros(shape=(iters, n))
        if current_option_number == 1:
            #  ------------------------Decision Trees Classifiers only------------------------------------
            self.stumps = np.zeros(shape=iters, dtype=object)
            self.stump_weights = np.zeros(shape=iters)
        #  ------------------------Decision Trees Classifiers only------------------------------------
        self.errors = np.zeros(shape=iters)

        """ Preparing data for Logistic Regression """
        X_manipulated = np.c_[np.ones((X.shape[0], 1)), X]
        y_manipulated = y[:, np.newaxis]
        theta = np.zeros((X_manipulated.shape[1], 1))
        self.sample_weights = np.zeros(shape=(iters, n))
        """ initialize weights uniformly """
        self.sample_weights[0] = np.ones(shape=n) / n

        for t in range(iters):
            # fit  weak learner
            curr_sample_weights = self.sample_weights[t]
            if current_option_number == 1:
                #  ------------------------Decision Trees Classifiers only------------------------------------
                stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
                stump = stump.fit(X, y, sample_weight=curr_sample_weights)

                # calculate error and stump weight from weak learner prediction
                stump_pred = stump.predict(X)
                err = curr_sample_weights[(stump_pred != y)].sum()  # / n
                stump_weight = np.log((1 - err) / err) / 2
                # update sample weights
                new_sample_weights = (
                        curr_sample_weights * np.exp(-stump_weight * y * stump_pred)
                )
                new_sample_weights /= new_sample_weights.sum()
                #  ------------------------Decision Trees Classifiers only------------------------------------
                # If not final iteration, update sample weights for t+1
                if t + 1 < iters:
                    self.sample_weights[t + 1] = new_sample_weights
                #  ------------------------Decision Trees Classifiers only------------------------------------
                # save results of iteration
                self.stumps[t] = stump
                self.stump_weights[t] = stump_weight
                # ------------------------Decision Trees Classifiers only------------------------------------
                self.errors[t] = err
        return self

    def predict(self, X):
        """ Make predictions using already fitted model """
        if current_option_number == 1:
            #  ------------------------Decision Trees Classifiers only------------------------------------
            stump_preds = np.array([stump.predict(X) for stump in self.stumps])
            return np.sign(np.dot(self.stump_weights, stump_preds))
            #  ------------------------Decision Trees Classifiers only------------------------------------


def plot_staged_adaboost(X, y, clf, errors, alphas, iters=10):
    """ Plot weak learner and cumulative strong learner at each iteration. """

    # larger grid
    fig, axes = plt.subplots(figsize=(8, iters * 3),
                             nrows=iters,
                             ncols=2,
                             sharex=True,
                             dpi=45)

    fig.set_facecolor('white')

    _ = fig.suptitle('Decision boundaries by iteration')
    for i in range(iters):
        ax1, ax2 = axes[i]

        if current_option_number == 1:
            #  ------------------------Decision Trees Classifiers only------------------------------------
            clf_i = clf.stumps[i]
            #  ------------------------Decision Trees Classifiers only------------------------------------
        elif current_option_number == 2:
            #  ------------------------Logistic Regression Classifiers only------------------------------------
            clf_i = clf.log_reg_clfs_and_weights[i]
            #  ------------------------Logistic Regression Classifiers only------------------------------------
        elif current_option_number == 3:
            # -------------------------naive_bayes classifiers---------------------------------------------
            clf_i = clf.naive_bayes_clfs[i]
            # -------------------------naive_bayes classifiers---------------------------------------------
        # Plot weak learner
        _ = ax1.set_title(f'Weak learner at t={i + 1}')
        plot_adaboost(X, y, clf_i,
                      sample_weights=clf.sample_weights[i],
                      annotate=False, ax=ax1, base_clf_errors=errors,
                      base_clf_stump_weights=alphas)
        # Plot strong learner
        trunc_clf = truncate_adaboost(clf, t=i + 1)
        _ = ax2.set_title(f'Strong learner at t={i + 1}')
        plot_adaboost(X, y, clf=trunc_clf,
                      sample_weights=clf.sample_weights[i],
                      annotate=False, ax=ax2, base_clf_errors=errors,
                      base_clf_stump_weights=alphas)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    """ Logistic Regression implementation """

    def __init__(self):
        self.activation_func = sigmoid

    def net_input(self, theta, _X):
        # Computes the weighted sum of inputs
        return np.dot(_X, theta)

    def probability(self, theta, _X):
        # Returns the probability after passing through sigmoid
        return self.activation_func((self.net_input(theta, _X)))

    def cost_function(self, theta, _X, _y, sample_weights):
        # Computes the cost function for all the training samples
        m = _X.shape[0]
        if all(sample_weight == 1 for sample_weight in sample_weights):
            current_samples_X = _X
        else:
            weighted_samples = np.concatenate((np.multiply(_X[:, 1], sample_weights)[:, np.newaxis],
                                               np.multiply(_X[:, 2], sample_weights)[:, np.newaxis]), axis=1)
            # weighted sample X
            current_samples_X = np.c_[np.ones((weighted_samples.shape[0], 1)), weighted_samples]

        total_cost = -(1 / m) * np.sum(
            _y * np.log(self.probability(theta, current_samples_X)) + (1 - _y) * np.log(
                1 - self.probability(theta, current_samples_X)
            )
        )
        return total_cost

    def gradient(self, theta, _X, _y, sample_weights):
        # Computes the gradient of the cost function at the point theta
        m = _X.shape[0]

        if all(sample_weight == 1 for sample_weight in sample_weights):
            current_samples_X = _X
        else:
            weighted_samples = np.concatenate((np.multiply(_X[:, 1], sample_weights)[:, np.newaxis],
                                               np.multiply(_X[:, 2], sample_weights)[:, np.newaxis]), axis=1)
            # weighted sample X
            current_samples_X = np.c_[np.ones((weighted_samples.shape[0], 1)), weighted_samples]

        return (1 / m) * np.dot(current_samples_X.T,
                                self.activation_func(self.net_input(theta, current_samples_X)) - _y)

    def fit(self, _X, _y, theta, sample_weights):
        opt_weights = fmin_tnc(func=self.cost_function, x0=theta,
                               fprime=self.gradient, args=(_X, _y.flatten(), sample_weights))
        return opt_weights[0]

    def predict(self, _X, parameters):
        theta = parameters[:, np.newaxis]
        return self.probability(theta, _X)

    def accuracy(self, _X, actual_classes, probab_threshold=0.5, parameters=None):
        predicted_classes = (self.predict(_X, parameters) >=
                             probab_threshold).astype(int)
        last_axis = len(predicted_classes.shape) - 1
        predicted_classes = np.squeeze(predicted_classes.astype(int), axis=last_axis)
        if last_axis < 2:
            predicted_classes = np.array(list(map(lambda pred: -1 if pred == 0 else 1, predicted_classes)))
        else:  # more than 2 dimensions - covers for now only 3-dimension vectors
            predicted_classes = np.array(list(map(lambda preds:
                                                  np.array(list(map(lambda pred: -1 if pred == 0 else 1, preds))),
                                                  predicted_classes)))
        accuracy = np.mean(predicted_classes == actual_classes)
        return accuracy * 100


# def run_adaboost():
#     scoring_videos_RDD = get_scoring_videos_RDD()
#     updated_videos_df = updated_videos_RDD.toDF(['id', 'bracket_count', 'contains_bad_language',
#                                                  'is_safe', 'final_score'])
# updated_videos_json = updated_videos_df.toJSON().collect()
#     X, y = get_mapped_transcript_dataset(updated_videos_json)
#     AdaBoost_instance = AdaBoost()
#     classifier = AdaBoost_instance.fit(X, y, iters=10)
#     pickle.dump(classifier, open(filename, 'wb'))
#     loaded_classifier = pickle.load(open(filename, 'rb'))
#     errors = AdaBoost_instance.errors
#     if current_option_number == 1:
#         #  ------------------------Decision Trees Classifiers only------------------------------------
#         # Base classifiers coefficient
#         stump_weights = AdaBoost_instance.stump_weights
#         plot_adaboost(X, y, classifier, base_clf_errors=errors, base_clf_stump_weights=stump_weights)
#         plot_staged_adaboost(X, y, classifier, errors, stump_weights)
#         #  ------------------------Decision Trees Classifiers only------------------------------------
#     elif current_option_number == 2:
#         #  ------------------------Logistic Regression Classifiers only------------------------------------
#         log_reg_weights_in_vote = AdaBoost_instance.log_reg_clfs_weights_in_vote
#         plot_adaboost(X, y, classifier, base_clf_errors=errors, base_clf_stump_weights=log_reg_weights_in_vote)
#         plot_staged_adaboost(X, y, classifier, errors, log_reg_weights_in_vote)
#         #  ------------------------Logistic Regression Classifiers only------------------------------------
#     elif current_option_number == 3:
#         # -------------------------naive_bayes Classifiers-------------------------------------
#         naive_bayes_clfs_weights = AdaBoost_instance.naive_bayes_clfs_weights
#         plot_adaboost(X, y, classifier, base_clf_errors=errors, base_clf_stump_weights=naive_bayes_clfs_weights)
#         plot_staged_adaboost(X, y, classifier, errors, naive_bayes_clfs_weights)
#         # -------------------------naive_bayes Classifiers-------------------------------------
#
#     train_err = (classifier.predict(X) != y).mean()
#     print(f'Train error: {train_err:.1%}')
#     return classifier


# def showMenu():
#     print("1) AdaBoost using Decision Trees Classifiers\n")
#     print("2) AdaBoost Using Linear Regression classifiers\n")
#     print("3) AdaBoost Using naive_bayes classifiers\n")
#     print("4) Exit\n")
#     option = int(input("Enter your choice: "))
#
#     return option


# current_option_number = showMenu()

# while current_option_number != 4:
#     if current_option_number == 1:
#         print("AdaBoost using Decision Trees Classifiers Execution:")
#         run_adaboost()
#
#     elif current_option_number == 2:
#         print("AdaBoost Using Linear Regression Classifiers Execution")
#         run_adaboost()
#     elif current_option_number == 3:
#         print("AdaBoost Using naive_bayes Classifiers Execution")
#         run_adaboost()
#     elif current_option_number == 4:
#         print("Exiting")
#     else:
#         print("'%s' is an unknown option." % current_option_number)
#     current_option_number = showMenu()
# -------------------------------Classifier--------------------------------------

# HTTP server for listening to the DB server and provide services to the DB server (as a client)
httpd = HTTPServer(('localhost', 8081), SimpleHTTPRequestHandler)
print("server is live\n")
httpd.serve_forever()
