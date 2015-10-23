import math
from decimal import Decimal
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    count = 0.0
    unigram_p = {}
    unigram_tuples = {}
    unigram_temp  = {}
    count_unigram = {}
    for lines in training_corpus:
        lines = lines + ' ' + STOP_SYMBOL
        words = lines.split()
        for word in words:
            count = count+1
            if (word,) in unigram_temp:
                unigram_temp[(word,)] += 1
            else:
                unigram_temp[(word,)] = 1
    unigram_tuples = unigram_temp


    bigram_p = {}
    prev_word = START_SYMBOL
    for lines in training_corpus:
        lines = lines + ' ' + STOP_SYMBOL
        words = lines.split()
        for word in words:
            bigram = (prev_word,word)
            if bigram in bigram_p:
                bigram_p[bigram] += 1.0
            else:
                bigram_p[bigram] = 1.0
            if word == STOP_SYMBOL:
                prev_word = START_SYMBOL
            else:
                prev_word = word

    word1 = START_SYMBOL
    word2 = START_SYMBOL
    trigram_p = {}
    for lines in training_corpus:
        lines =  lines + ' '+ STOP_SYMBOL
        words = lines.split()
        for word in words:
            trigram = (word1, word2, word)
            if trigram in trigram_p:
                trigram_p[trigram] += 1.0
            else:
                trigram_p[trigram] = 1.0
            if word == STOP_SYMBOL:
                word1 = START_SYMBOL
                word2 = START_SYMBOL
            else:
                word1 = word2
                word2 = word

    for item in trigram_p:
        x = (item[0],item[1],)
        if x in bigram_p:
            count_prev = bigram_p[x]
            trigram_p[item] = math.log(trigram_p[item]/count_prev,2)
        elif x == (START_SYMBOL,START_SYMBOL):
            trigram_p[item] = math.log(trigram_p[item]/len(training_corpus),2)


    for item in bigram_p:
        x = (item[0],)
        if x in unigram_temp:
            count_prev = unigram_temp[x]
            bigram_p[item] = math.log(bigram_p[item]/count_prev ,2)
        elif item[0] == START_SYMBOL:
            bigram_p[item] = math.log(bigram_p[item]/len(training_corpus) ,2)


    for item in unigram_tuples:
        unigram_tuples[item] = math.log(unigram_tuples[item]/count,2)

    unigram_p = unigram_tuples

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc.
def score(ngram_p, n, corpus):
    sen_scores = 0
    scores = []
    if n ==1:
        for lines in corpus:
            lines = lines + ' '+ STOP_SYMBOL
            words = lines.split()
            for word in words:
                if (word,) in ngram_p:
                    sen_scores = sen_scores + ngram_p[(word,)]
                    if word == STOP_SYMBOL:
                        scores.append(sen_scores)
                        sen_scores = 0
                else:
                    sen_scores = MINUS_INFINITY_SENTENCE_LOG_PROB
                    scores.append(sen_scores)
                    sen_scores = 0
                    break

    prev_word = START_SYMBOL
    sen_scores = 0
    if n ==2:
        scores = []
        for lines in corpus:
            lines = lines + ' ' + STOP_SYMBOL
            words = lines.split()
            for word in words:
                bigram =  (prev_word,word)
                if bigram in ngram_p:

                    sen_scores = sen_scores + ngram_p[bigram]
                    if word == STOP_SYMBOL:
                        scores.append(sen_scores)
                        prev_word = START_SYMBOL
                        sen_scores = 0
                    else:
                        prev_word = word
                else:
                    sen_scores = MINUS_INFINITY_SENTENCE_LOG_PROB
                    scores.append(sen_scores)
                    sen_scores = 0
                    prev_word = START_SYMBOL
                    break

    word1 = START_SYMBOL
    word2 = START_SYMBOL

    if n ==3:
        scores = []
        for lines in corpus:
            lines =  lines + ' ' + STOP_SYMBOL
            words = lines.split()
            for word in words:
                if (word1,word2,word) in ngram_p:
                    sen_scores =  sen_scores + ngram_p[(word1,word2,word)]
                    if word == STOP_SYMBOL:
                        scores.append(sen_scores)
                        sen_scores = 0
                        word1 = START_SYMBOL
                        word2 = START_SYMBOL
                    else:
                        word1 = word2
                        word2 = word
                else:
                    sen_scores = MINUS_INFINITY_SENTENCE_LOG_PROB
                    scores.append(sen_scores)
                    sen_scores = 0
                    word1 = START_SYMBOL
                    word2 = START_SYMBOL
                    break


    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):

    scores = []
    sen_score = 0
    word1 = START_SYMBOL
    word2 = START_SYMBOL
    lambda1 = math.log(1,2) - math.log(3,2)
    for lines in corpus:
        lines = lines + ' ' + STOP_SYMBOL
        words = lines.split()
        for word in words:
            if (word1,word2,word) not in trigrams and (word2,word) not in bigrams and (word,) not in unigrams:
                sen_score = MINUS_INFINITY_SENTENCE_LOG_PROB
                scores.append(sen_score)
                sen_score = 0
                word1 = START_SYMBOL
                word2 = START_SYMBOL
                break
            if (word1,word2,word) in trigrams:
                tri_prob = math.exp(trigrams[(word1,word2,word)] * math.log(2))
                bi_prob = math.exp(bigrams[(word2,word)] * math.log(2))
                uni_prob = math.exp(unigrams[(word,)] * math.log(2))

                sen_score = sen_score + (lambda1 + math.log((uni_prob + bi_prob + tri_prob),2))
            elif (word2,word) in bigrams:
                bi_prob = math.exp(bigrams[(word2,word)] * math.log(2))
                uni_prob = math.exp(unigrams[(word,)] * math.log(2))
                sen_score = sen_score + (lambda1 + math.log((uni_prob + bi_prob),2))
            elif (word,) in unigrams:
                uni_prob = math.exp(unigrams[(word,)] * math.log(2))
                sen_score = sen_score + (lambda1 + math.log((uni_prob),2))
            else:
                sen_score = MINUS_INFINITY_SENTENCE_LOG_PROB
            if word == STOP_SYMBOL:
                scores.append(sen_score)
                sen_score = 0
                word1 = START_SYMBOL
                word2 = START_SYMBOL
            else:
                word1 = word2
                word2 = word


    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()