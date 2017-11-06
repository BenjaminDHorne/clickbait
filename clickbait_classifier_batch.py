from sklearn import preprocessing,svm
from sklearn.pipeline import Pipeline
import nltk
import numpy
import subprocess
import time
import utility
import os
import json
import unicodedata

###############  Classifier  ############################
print "creating classifier..."
no_samples = 10000
positive = numpy.loadtxt("vectors/positive.csv",  delimiter=',')
negetive = numpy.loadtxt("vectors/negative.csv", delimiter=',')
X = numpy.concatenate((positive, negetive), axis=0)
p = numpy.ones((no_samples, 1))
n = numpy.full((no_samples, 1), -1, dtype=numpy.int64)
Y = numpy.concatenate((p,n), axis=0)
y = Y.ravel()
#scale
scaler = preprocessing.StandardScaler().fit(X)
#classifying
svm_module = svm.SVC()
classifier = Pipeline(steps= [('svm', svm_module)]) #[('scale', scaler), ('svm', svm_module)])
classifier.fit(X, y)
print "Classifer created"
##############################################################

def fix(text): #use this method to fix any json errors
    try:
        text = text.decode("ascii", "ignore")
    except:
        t=[unicodedata.normalize('NFKD', unicode(q)).encode('ascii','ignore') for q in text]
        text=''.join(t).strip()
    return text

##################### Clickbait Classifier Service ###########################
def isclickbait(document):
	try:
		title_vector = numpy.array(utility.create_vector(document)).reshape(1,-1)
		# t = scaler.transform(title_vector)
		prediction = classifier.predict(title_vector)
		if prediction[0] == 1:
			return 1
		else:
			return 0
	except Exception,e:
		print "except:", e

if __name__ == '__main__':
	#Batch File Version
	outfile = "../clickbait_features.csv"
	with open(outfile, "a") as out:
		out.write("id,clickbait\n")
	path = ""
	for dirName, subdirList, fileList in os.walk(path):
		for fn in fileList:
			source = dirName.split("/")[-1]
			id = fn.split(".")[0]
			with open(dirName+"/"+fn) as json_file:
				x = json.loads(json_file.readline())
				title_text = x['title']
				title_text = fix(title_text.strip())
				clickbait = isclickbait(title_text)
			with open(outfile, "a") as out:
				out.write(",".join((id,str(clickbait)))+"\n")
