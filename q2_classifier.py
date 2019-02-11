import numpy as np
import sys
import math


def test(test_file, out_file, uniquewords, p_ham, p_spam):
    test_fd = open(test_file, 'r')
    out_fd = open(out_file, 'w')
    num_correct = 0
    count=0
    #taking logs of probabilities of ham and spam
    logp_ham= np.log10(p_ham)
    logp_spam=np.log10(p_spam)
    #count is maintained because we need total number of words to calculate accuracy
    for doc in test_fd:
        list_of_words_doc = doc.strip().split()
        prob_ham = 0.0
        prob_spam = 0.0
        i = 2
        while (i < len(list_of_words_doc)):
            if list_of_words_doc[i] in uniquewords:
                prob_ham += int(list_of_words_doc[i + 1]) * logp_ham[uniquewords[list_of_words_doc[i]]]
                prob_spam += int(list_of_words_doc[i + 1]) * logp_spam[uniquewords[list_of_words_doc[i]]]
                pass
            else:
                print "new word ", list_of_words_doc[i]
            i += 2

        # classify ham/spam based on training data
        if prob_ham > prob_spam:
            # test_ham_count += 1
            out_fd.write(list_of_words_doc[0] + ' ' + "ham\n")
            count +=1
            if (list_of_words_doc[1] == 'ham'):
                num_correct += 1
        elif prob_spam > prob_ham:
            # test_spam_count += 1
            out_fd.write(list_of_words_doc[0] + ' ' + "spam\n")
            count+=1
            if (list_of_words_doc[1] == 'spam'):
                num_correct += 1
            else :num_correct = num_correct -1
        else:
            print("equal probabilities")
            num_correct += 1
            count+=1
    print "Accuracy  : ", float(num_correct) / count
    out_fd.close()
    test_fd.close()


def calculate_pham_pspam(count_uniquewords_docs, ham_rows, spam_rows):
    # count occurences of each unique word in spam/ham
    uniquewords_count_ham = np.sum(count_uniquewords_docs[ham_rows], 0)

    zero_count_words_indices_ham = np.where(uniquewords_count_ham == 0)[0]
    # print len(zero_count_words_indices_ham)
    uniquewords_count_ham[zero_count_words_indices_ham] += 0.00000001

    uniquewords_count_spam = np.sum(count_uniquewords_docs[spam_rows], 0)
    zero_count_words_indices_spam = np.where(uniquewords_count_spam == 0)[0]

    uniquewords_count_spam[zero_count_words_indices_spam] += 0.00000001

    total_words_in_ham_docs = np.sum(uniquewords_count_ham)
    total_words_in_spam_docs = np.sum(uniquewords_count_spam)
     #returning probabilities of ham and spam
    return(uniquewords_count_ham / total_words_in_ham_docs, uniquewords_count_spam / total_words_in_spam_docs)


def unique_wordcount(train_file, uniquewords, ham_rows, spam_rows):
    train_fd = open(train_file, 'r')
    num_hams = len(ham_rows)
    num_spams = len(spam_rows)
    count_uniquewords_docs = np.zeros((num_hams + num_spams, len(uniquewords)))
    doc_num = 0
    for doc in train_fd:
        list_of_words_doc = doc.strip().split()
        i = 2
        while (i < len(list_of_words_doc)):
            count_uniquewords_docs[doc_num][uniquewords[list_of_words_doc[i]]] += int(list_of_words_doc[i + 1])
            i += 2
        doc_num += 1
    train_fd.close()
    return count_uniquewords_docs


def find_uniquewords_and_docs_rows(train_file):
    uniquewords = dict()

    train_fd = open(train_file, 'r')
    count_of_uniquewords = 0
    doc_num = 0

    ham_rows = []
    spam_rows = []
    for doc in train_fd:
        list_of_words_doc = doc.strip().split()
        i = 2
        while (i < len(list_of_words_doc)):
            if list_of_words_doc[i] not in uniquewords:
                uniquewords[list_of_words_doc[i]] = count_of_uniquewords
                count_of_uniquewords += 1
            i += 2
        if list_of_words_doc[1] == 'ham':
            ham_rows.append(doc_num)
        elif list_of_words_doc[1] == 'spam':
            spam_rows.append(doc_num)
        else:
            print "unknown doc class"
        doc_num += 1

    train_fd.close()


    return uniquewords, ham_rows, spam_rows


def train(train_file):
   
    uniquewords, ham_rows, spam_rows = find_uniquewords_and_docs_rows(train_file)
  
    docs_words_count = unique_wordcount(train_file, uniquewords, ham_rows, spam_rows)

    p_ham, p_spam = calculate_pham_pspam(docs_words_count, ham_rows, spam_rows)

    # return uniquewords, logp_ham, logp_spam
    return uniquewords,p_ham, p_spam


def main():
    # print "This is the name of the script: ", sys.argv[0]
    # print "Number of arguments: ", len(sys.argv)
    # print "The arguments are: " , str(sys.argv)
    if len(sys.argv) < 7:
        print "not enough arguments"
        return
    elif len(sys.argv) > 7:
        print "too many arugments"
        return
    if (sys.argv[1] == "-f1"):
        train_file = sys.argv[2]
    else:
        print "check arguments"
        return
    if (sys.argv[3] == "-f2"):
        test_file = sys.argv[4]
    else:
        print "check arguments"
        return
    if (sys.argv[5] == "-o"):
        out_file = sys.argv[6]
    else:
        print "check arguments"
        return

    print train_file, test_file, out_file

    # training 
    uniquewords, pham, pspam = train(train_file)
    # testing , pass list of unique words
    test(test_file, out_file, uniquewords, pham, pspam)


if __name__ == '__main__':
    main()