#this code creates training and test folders
#training files contain 60percent of the corresponding author's texts
#text files contain the remaining 40 percent 
import os
from shutil import copyfile
from random import shuffle


directory = raw_input('Enter the directory of the data: ')

authorlist = []
for filename in os.listdir(directory):
	authorlist.append(filename)
for author in authorlist:
	if author == '.DS_Store':
		authorlist.remove('.DS_Store')


#creating new directories for training and test
if os.path.isdir('./training'):
	pass
else:
	os.mkdir('./training')

if os.path.isdir('./test'):
	pass
else:
	os.mkdir('./test')

#creating author folders in the training and test folders
for author in authorlist:
	if os.path.isdir('./training/'+str(author)):
		pass
	else:
		os.mkdir('./training/'+str(author))
	if os.path.isdir('./test/'+str(author)):
		pass
	else:
		os.mkdir('./test/'+str(author))

generalpath = os.getcwd()

#copying author texts randomly to new training and test folders
for author in authorlist:
	directory_author = directory+'/'+str(author)
	textlist = []
	for filename in os.listdir(directory_author):
		textlist.append(filename)
	for text in textlist:
		if text == '.DS_Store':
			textlist.remove('.DS_Store')
	shuffle(textlist)
	count60perc = int(len(textlist)*0.6)
	count = 0
	for text in textlist:
		src = directory_author+'/'+str(text)
		if count < count60perc:
			dst = generalpath+'/training/'+str(author)+'/'+str(text)
		else:
			dst = generalpath+'/test/'+str(author)+'/'+str(text)
		copyfile(src,dst)
		count = count + 1

