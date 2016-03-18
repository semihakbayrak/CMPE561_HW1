#-*- coding: utf-8 -*- 
import os
import re
import locale
from collections import Counter
import numpy as np


#fileName = '/Users/semihakbayrak/Dersler/CMPE561/HW1/training'



#input from the user
directory_train = raw_input('Enter the directory of the training data: ')
directory_test = raw_input('Enter the directory of the test data: ')

#authors
authorlist = []
author_vocabularies = {}
for filename in os.listdir(directory_train):
	authorlist.append(filename)
for author in authorlist:
	if author == '.DS_Store':
		authorlist.remove('.DS_Store')

priors = np.array([]) 
conclistall = []
for author in authorlist:
	directory_author = directory_train+'/'+str(author)
	textlist = []
	
	for filename in os.listdir(directory_author):
		textlist.append(filename)
	
	for text in textlist:
		if text == '.DS_Store':
			textlist.remove('.DS_Store')
	
	priors = np.append(priors,1.0*len(textlist)) #prior probabilities for authors
	conclist1 = []
	
	#tokenization and training
	for text in textlist:
		directory_text = directory_author+'/'+str(text)
		textfile = open(directory_text,'r').read()
		tok = textfile.decode('iso-8859-9')
		tok2 = tok.lower()
		tok3 = re.sub(r"\d+",'',tok2,flags=re.U)    #remove numbers
		tok4 = re.sub(r"\W+",'\n',tok3,flags=re.U)  #remove non alphanumerics with new line
		conclist2 = tok4.split()
		conclist1 = conclist1 + conclist2 #all the words used by author in a list

	author_vocabularies[str(author)]=Counter(conclist1) #word frequencies for authors in dictionaries
	conclistall = conclistall + conclist1 #all the words used by all authors in a list

vocabulary = Counter(conclistall) #vocabulary for words with their frequencies in a dictionary
priors = priors/sum(priors) #prior normalization

#number of words used by authors
wordcountsauthor = []
for i in range(len(authorlist)):
	wordcountsauthor.append(sum(author_vocabularies[str(authorlist[i])].values()))

counttrue = 0
countfalse = 0

#will be used for precision and recall calculation
RecallTrue = {}
RecallFalse = {}
PrecisionTrue = {}
PrecisionFalse = {}
for author in authorlist:	
	RecallFalse[str(author)] = 0
	RecallTrue[str(author)] = 0
	PrecisionFalse[str(author)] = 0
	PrecisionTrue[str(author)] = 0

for author in authorlist:
	directory_author = directory_test+'/'+str(author)
	textlist = []
	for filename in os.listdir(directory_author):
		textlist.append(filename)
	for text in textlist:
		if text == '.DS_Store':
			textlist.remove('.DS_Store')
	for text in textlist:
		#tokenization for test inputs
		directory_text = directory_author+'/'+str(text)
		textfile = open(directory_text,'r').read()
		tok = textfile.decode('iso-8859-9')
		tok2 = tok.lower()
		tok3 = re.sub(r"\d+",'',tok2,flags=re.U)
		tok4 = re.sub(r"\W+",'\n',tok3,flags=re.U)
		tlist = tok4.split()
		#multinomial naive bayes with laplace smoothing
		logprob = np.log2(priors)
		alfa = 0.01 
		for word in tlist:
			for i in range(len(authorlist)):
				if word in author_vocabularies[str(authorlist[i])]:
					probw = 1.0*(author_vocabularies[str(authorlist[i])][word]+alfa)/(wordcountsauthor[i]+alfa*len(vocabulary))
				else:
					probw = 1.0*alfa/(wordcountsauthor[i]+alfa*len(vocabulary))
				logprob[i] = logprob[i] + np.log2(probw)
		print authorlist[np.argmax(logprob)]
		#Precision and Recall calculation
		if authorlist[np.argmax(logprob)]==author:
			print '1'
			counttrue = counttrue + 1
			RecallTrue[str(author)] = RecallTrue[str(author)] + 1             #True Pozitive
			PrecisionTrue[str(author)] = PrecisionTrue[str(author)] + 1       #True Pozitive
		else:
			print '0'
			countfalse = countfalse + 1
			RecallFalse[str(author)] = RecallFalse[str(author)] + 1                                                            #False Negative
			PrecisionFalse[str(authorlist[np.argmax(logprob)])] = PrecisionFalse[str(authorlist[np.argmax(logprob)])] + 1      #False Pozitive
	

#Precision and Recall calculation
RT_list = np.zeros((len(authorlist)))
RF_list = np.zeros((len(authorlist)))
PT_list = np.zeros((len(authorlist)))
PF_list = np.zeros((len(authorlist)))
count = 0
for author in authorlist:
	RT_list[count] = 1.0*RecallTrue[str(author)]
	RF_list[count] = 1.0*RecallFalse[str(author)]
	PT_list[count] = 1.0*PrecisionTrue[str(author)]
	PF_list[count] = 1.0*PrecisionFalse[str(author)]
	count = count + 1

with np.errstate(divide='ignore', invalid='ignore'):
    Precision = PT_list/(PT_list+PF_list)    #in the case of no true or false assignment to one author, I will encounter 0/0 problem
    Precision[(PT_list+PF_list) == 0] = 0    #I used this to overcome 0 division problem
Recall = RT_list/(RT_list+RF_list)
macroPrecision = sum(Precision)/len(Precision)
macroRecall = sum(Recall)/len(Recall)
microPrecision = sum(PT_list)/sum(PT_list+PF_list)
microRecall = sum(RT_list)/sum(RT_list+RF_list)
macroF = 2*macroPrecision*macroRecall/(macroPrecision+macroRecall)
microF = 2*microPrecision*microRecall/(microPrecision+microRecall)
print "Macro Precision = %f" % macroPrecision
print "Micro Precision = %f" % microPrecision
print "Macro Recall = %f" % macroRecall
print "Micro Recall = %f" % microRecall
print "Macro F-score = %f" % macroF
print "Micro F-score = %f" % microF
accuracy = 1.0*counttrue/(counttrue+countfalse)
print accuracy


