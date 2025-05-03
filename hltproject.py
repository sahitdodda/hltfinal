import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('gutenberg')

from nltk.corpus import gutenberg

print(gutenberg.fileids())
text = gutenberg.raw('austen-emma.txt')

