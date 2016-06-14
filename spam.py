import os
from os import listdir
from os.path import isfile, join
from math import log, exp

DEBUG = False


class Probabilities:
    def __init__(self, labels=None, counts=None):
        if labels is None:
            labels = []
        if counts is None:
            counts = []

        # Check there's a 1-to-1 correspondance between labels and counts
        if len(labels) != len(counts):
            raise Exception("The number of labels is different than the number of counts")

        self.total = sum(counts)
        self.num_categories = len(labels)
        self.label_count = dict(zip(labels, counts))

    def probabilty(self, label):
        """
        Given:
            string label: the label

        Output:
            float probability: the probability of getting the label
        """
        if label not in self.label_count:
            raise Exception("Unknown probability.")

        return self.label_count[label] / self.total

    def smooth_probability(self, label, k=1):
        """
        n = total number of samples
        d = number of categories
        x = N(elem) = number of elements of category

        The original result would simply be x / n.
        However, to smooth things, we add k samples to everything in our initial counts
        and then compute the probability (x + k) / (n + k*d)

        Given:
            string label: the label

        Output:
            float probability: the probability of getting the label, calculated with Laplace Smoothing.
        """
        num_occurences = self.label_count.get(label, 0)
        return (num_occurences + k) / (self.total + k * self.num_categories)

    def has_word(self, word):
        return word not in self.label_count

    def add_occurence(self, label, num=1):
        """
        Adds n occurents of events of type label to our counts.
        """
        if label not in self.label_count:
            self.num_categories += 1
            self.label_count[label] = num
        else:
            self.label_count[label] += num

        self.total += num


class SpamProbabilities:
    def __init__(self, words=None, counts_spam=None, counts_ham=None):
        if words is None:
            words = []
        if counts_spam is None:
            counts_spam = []
        if counts_ham is None:
            counts_ham = []

        if len(words) != len(counts_spam) or len(words) != len(counts_ham):
            raise Exception("There is not a 1-to-1 correspondance between words and counts.")

        self.prob_spam = Probabilities(words, counts_spam)
        self.prob_ham = Probabilities(words, counts_ham)

    def prob_word_is_spam(self, word):
        # Pr(S | W) = P(W | S) * P(S) / ((P(W | S) * P(S) + P(W | H) * P(H))
        p_is_spam = self.prob_spam.smooth_probability(word)
        p_is_ham = self.prob_ham.smooth_probability(word)

        num_spam = self.prob_spam.total
        num_ham = self.prob_ham.total
        total_num = num_ham + num_spam

        p_spam = num_spam / total_num
        p_ham = num_ham / total_num

        return (p_is_spam * p_spam) / (p_is_spam * p_spam + p_is_ham * p_ham)

    def prob_message_are_spam(self, words):
        """
        To check whether a list of words is spam, we use Baysean theory.

        P(Spam | words) = P(words | Spam) * P(Spam) / P(words)
                        = P(words | Spam) * P(Spam) / (P(words | Spam) * P(Spam) + P(words | not Spam) * P(not Spam))
        P(words | Spam) = product(P(word | Spam) for word in words)
        P(words) = sum(P(word|Spam)*P(Spam) + P(word|not Spam) * P(not Spam) for word in words)

        We'll use the simplifcation in Wikipedia's page: P(words | Spam) / (P(words | Spam) + P(words | not Spam))
        TODO: see if the original formula is better, and if so use it.
        """
        n = 0
        for word in words:
            p = self.prob_word_is_spam(word)
            if DEBUG:
                print("Word: ", word)
                print("Probability of word being spammy:", p)
                print("n: ", n)
                print("log(1 - p):", log(1 - p))
                print("log(p): ", -log(p))
            n += log(1 - p) - log(p)

        if DEBUG:
            print(n)

        if n > 200:
            return 0
        elif n < -100:
            return 1
        else:
            return 1 / (1 + exp(n))

    def add_spam_occurences(self, words):
        if type(words) == str:
            words = words.split()

        for word in words:
            self.prob_spam.add_occurence(word)

    def add_ham_occurences(self, words):
        if type(words) == str:
            words = words.split()

        for word in words:
            self.prob_ham.add_occurence(word)


def get_filenames_in_directory(dir_name):
    return [os.path.join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]


def filter_words(words):
    result = []
    for word in words:
        if len(word) <= 3:
            continue

        result.append(word)
    return result


def get_words(f):
    result = []
    for line in f:
        result.extend(line.split())
    return filter_words(result)


def train(s, spam_files_dir, ham_files_dir):
    for filename in get_filenames_in_directory(spam_files_dir):
        with open(filename) as f:
            words = get_words(f)
            s.add_spam_occurences(words)

    for filename in get_filenames_in_directory(ham_files_dir):
        with open(filename) as f:
            words = get_words(f)
            s.add_ham_occurences(words)


SPAM_THRESHOLD = 0.98


def test(s, spam_files_dir, ham_files_dir):
    spam_file_names = get_filenames_in_directory(spam_files_dir)
    ham_file_names = get_filenames_in_directory(ham_files_dir)

    total_seen = 0
    incorrect = 0

    for spam_filename in spam_file_names[:]:
        with open(spam_filename) as f:
            words = get_words(f)
            prob = s.prob_message_are_spam(words)

            if prob < SPAM_THRESHOLD:
                incorrect += 1
                print("Non-detected spam: ", ' '.join(words), '\n\n')
            total_seen += 1

    for ham_filename in ham_file_names[:]:
        with open(ham_filename) as f:
            words = get_words(f)
            prob = s.prob_message_are_spam(words)

            if prob > SPAM_THRESHOLD:
                incorrect += 1
                print("Detected non-spam: ", ' '.join(words), '\n\n')
            total_seen += 1

    failure_rate = incorrect / total_seen
    print("Total seen: ", total_seen)
    print("Total wrong: ", incorrect)
    print("Failure rate: %.1f%%" % (failure_rate * 100))


SPAM_DIR_TEST = 'spam-test'
HAM_DIR_TEST = 'nonspam-test'

SPAM_DIR_TRAIN = 'spam-train'
HAM_DIR_TRAIN = 'nonspam-train'

if __name__ == '__main__':
    s = SpamProbabilities()
    train(s, SPAM_DIR_TRAIN, HAM_DIR_TRAIN)
    test(s, SPAM_DIR_TEST, HAM_DIR_TEST)
