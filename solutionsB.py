import sys
import nltk
import math
import time
import re
import copy

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    word_list = []
    tag_list = []
    for lines in  brown_train:
        lines  = START_SYMBOL + ' ' + lines + ' ' + STOP_SYMBOL
        lines  = lines.split()
        for words in lines:
            x = words.split('/')

            if len(x) > 0:
                if x[0] == START_SYMBOL:
                    word_list.append(START_SYMBOL)
                    word_list.append(START_SYMBOL)
                    tag_list.append(START_SYMBOL)
                    tag_list.append(START_SYMBOL)
                elif x[0] == STOP_SYMBOL:
                    word_list.append(x[0])
                    tag_list.append(x[0])
                    brown_words.append(word_list)
                    brown_tags.append(tag_list)
                    tag_list = []
                    word_list = []
                else:
                    word_list.append(x[0])
                    if len(x)==2:
                        tag_list.append(x[1])

    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    bigrams = {}
    prev_word = START_SYMBOL
    for items in brown_tags:
        for word in items:
            if word == START_SYMBOL:
                continue
            if word == STOP_SYMBOL:
                bigram = (prev_word,word)
                if bigram in bigrams:
                    bigrams[bigram] +=1.0
                else:
                    bigrams[bigram] = 1.0
                prev_word = START_SYMBOL
            else:
                bigram = (prev_word,word)
                if bigram in bigrams:
                    bigrams[bigram] +=1.0
                else:
                    bigrams[bigram] = 1.0
                prev_word = word

    word1 =START_SYMBOL
    word2 = START_SYMBOL
    for items in brown_tags:
        for word in items:
            if word == START_SYMBOL:
                continue
            if word == STOP_SYMBOL:
                trigram = (word1,word2,word)
                if trigram in q_values:
                    q_values[trigram] +=1.0
                else:
                    q_values[trigram] = 1.0
                word1 = START_SYMBOL
                word2 = START_SYMBOL
            else:
                trigram = (word1,word2,word)
                if trigram in q_values:
                    q_values[trigram] +=1.0
                else:
                    q_values[trigram] = 1.0
                word1 =word2
                word2 = word

    for items in q_values:
        x = (items[0],items[1])
        if x in bigrams:
            q_values[items] = math.log(q_values[items]/bigrams[x],2)
        elif x ==  (START_SYMBOL,START_SYMBOL):
            q_values[items] = math.log(q_values[items]/len(brown_tags),2)


    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    map = {}
    for item in brown_words:
        for words in item:
            if words in map:
                map[words] +=1
            else:
                map[words] = 1

    known_words = set([])
    for items in map:
        if map[items] > RARE_WORD_MAX_FREQ:
            known_words.add(items)
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    temp_list = []
    brown_words_rare = []
    map = {}
    for i in known_words:
        if i in map:
            map[i] +=1
        else:
            map[i] = 1

    for lines in brown_words:
        for items in lines:
            if items == STOP_SYMBOL:
                if items in map:
                    temp_list.append(items)
                else:
                     temp_list.append(RARE_SYMBOL)
                brown_words_rare.append(temp_list)
                temp_list = []
            else:
                 if items in map:
                     temp_list.append(items)
                 else:
                    temp_list.append(RARE_SYMBOL)


    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    emission_count = {}
    count_tags = {}
    e_values = {}
    taglist = set([])

    for w,t in zip(brown_words_rare,brown_tags):
        for x,y in zip(w,t):
            if (x,y) in emission_count:
                emission_count[(x,y)] +=1
            else:
                emission_count[(x,y)] = 1


    for lines in brown_tags:
        for items in lines:
            if items in count_tags:
                count_tags[items] += 1.0
            else:
                count_tags[items] = 1.0

    for items in emission_count:
        e_values[items] = math.log(emission_count[items]/count_tags[items[1]],2)

    for items in count_tags:
        taglist.add(items)
    
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    temp = []
    taglist.remove(START_SYMBOL)
    taglist.remove(STOP_SYMBOL)
    string = ''
    for items in brown_dev_words:
        items.insert(0,START_SYMBOL)
        items.insert(0,START_SYMBOL)


        output_tags = calc_tags(items,taglist, known_words, q_values, e_values)

        for idx in range(2,len(items)):
            if string == '':
                string  = str(items[idx]) +'/'+str(output_tags[idx])
            else:
                string = string + ' ' +str(items[idx]) +'/'+str(output_tags[idx])
        string = string + '\n'
        tagged.append(string)
        string = ''




    return tagged

def calc_tags(sentence, taglist, known_words, q_values, e_values):

    items = copy.copy(sentence)
    pi = {}
    bp = {}
    pi[(1,START_SYMBOL,START_SYMBOL)] = 1
    for k in range(2,len(items)):
        if items[k] not in known_words:
            items[k] = RARE_SYMBOL
        w_tag_set = get_tag_set(k-2,taglist)
        u_tag_set = get_tag_set(k-1,taglist)
        v_tag_set = get_tag_set(k,taglist)

        for u in u_tag_set:
            for v in v_tag_set:
                p_max = LOG_PROB_OF_ZERO
                tag_max = '0'
                for w in w_tag_set:
                    if(w,u,v) in q_values and (items[k],v) in e_values:
                        p_val = pi[(k-1,w,u)] + q_values[(w,u,v)]+ e_values[(items[k],v)]
                    else:
                        p_val = LOG_PROB_OF_ZERO
                    if p_val> p_max:
                        p_max = p_val
                        tag_max = w
                pi[(k,u,v)] = p_max
                bp[(k,u,v)] = tag_max

    n = len(items)
    output_tags = [None]*n
    output_tags[0] = output_tags[1] = START_SYMBOL
    p_max = LOG_PROB_OF_ZERO
    u_tag_set = get_tag_set(n-2,taglist)
    v_tag_set = get_tag_set(n-1,taglist)
    for u in u_tag_set:
        for v in v_tag_set:
            if (u,v,"STOP") in q_values:
                p_val = pi[(n-1,u,v)] + q_values[(u,v,"STOP")]
            else:
                p_val = LOG_PROB_OF_ZERO
            if p_val > p_max:

                output_tags[n-1] = v
                output_tags[n-2] = u
                p_max = pi[(n-1,u,v)]

    for k in range(n-3,1,-1):
        if (k+2,output_tags[k+1],output_tags[k+2]) in bp:
            output_tags[k] = bp[(k+2,output_tags[k+1],output_tags[k+2])]
        else:
            output_tags[k] = 'NOUN'

    return output_tags

def get_tag_set(n,tag_set):
    # returns the set of possible tags for the word at index n in the sentence
    tag_set_star = set([START_SYMBOL])
    if n == 0 or n == 1:
        # word is *, so possible tag is only *
        return tag_set_star
    else:
        # normal word in the sentence, all tags possible
        return tag_set

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]
    tagged = []
    string = ''
    # IMPLEMENT THE REST OF THE FUNCTION HERE
    t0 = nltk.DefaultTagger('NOUN')
    t1 = nltk.UnigramTagger(training,backoff=t0)
    t2 = nltk.BigramTagger(training,backoff=t1)
    nltk_trigram = nltk.TrigramTagger(training,backoff=t2)
    for lines in brown_dev_words:

        x= nltk_trigram.tag(lines)
        for idx in range(0,len(x)):
            if string == '':
                string = str(x[idx][0]) + '/' + str(x[idx][1])
            else:
                string = string + ' ' + str(x[idx][0]) + '/' + str(x[idx][1])
        string = string + '\n'
        tagged.append(string)
        string = ''

    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "train.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
