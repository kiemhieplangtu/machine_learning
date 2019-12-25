import sys
import numpy as np
import tensorflow as tf

from tensorflow import keras


def decode_txt(txt):
	return ' '.join( [word_ind.get(i, '?') for i in txt] )


def txt_encode(txt, word_ind):
	ret = [1]

	for word in txt:
		word = word.lower()
		if( word in word_ind):
			ret.append( word_ind[word] )
		else:
			ret.append( 2 )

	return ret





## ---  MAIN --- ##

data = keras.datasets.imdb

(train_x, train_y), (test_x, test_y) = data.load_data( num_words = 88000 )


word_ind = data.get_word_index()
word_ind = { k:(v+3) for k,v in word_ind.items()}

word_ind['<PAD>']    = 0
word_ind['<START>']  = 1
word_ind['<UNK>']    = 2
word_ind['<UNUSED>'] = 3



train_x = keras.preprocessing.sequence.pad_sequences( train_x, value=word_ind['<PAD>'], padding='post', maxlen=350 )
test_x  = keras.preprocessing.sequence.pad_sequences( test_x, value=word_ind['<PAD>'], padding='post', maxlen=350 )

rev_word_ind = dict( [v, k] for (k,v) in word_ind.items() )

if(False):
	## model here
	model = keras.Sequential()
	model.add( keras.layers.Embedding(88000, 16) )
	model.add( keras.layers.GlobalAveragePooling1D() )
	model.add( keras.layers.Dense( 16, activation='relu' ) )
	model.add( keras.layers.Dense( 1, activation='sigmoid') )

	model.summary()

	model.compile( optimizer='adam',  loss='binary_crossentropy', metrics=['accuracy'] )

	x       = train_x[10000:]
	y       = train_y[10000:]
	valid_x = train_x[:10000]
	valid_y = train_y[:10000]

	fit = model.fit( x, y, epochs=40, batch_size=512, validation_data=(valid_x, valid_y), verbose=True)

	res = model.evaluate( test_x, test_y )

	print( res)

	model.save( 'movie_review.h5' )

	test_review = test_x[0]
	pred = model.predict( [test_review] )
	print(' Review: ')
	print( decode_txt(test_review) )
	print( 'Prediction: ' + str(pred[0] ))
	print( 'Actual: ' + str(test_y[0]) )

	sys.exit()


model = keras.models.load_model( 'movie_review.h5' )

with open('movie_review.txt', encoding='utf-8') as f:
	for line in f.readlines():
		xline = line.replace(',','').replace( '.', '' ).replace( '(', '' ).replace( ')', '' ).replace( ':', '' ).replace( '\"', '' ).strip().split(' ')
		encode = txt_encode(xline, word_ind)
		encode = keras.preprocessing.sequence.pad_sequences( [encode], value=word_ind['<PAD>'], padding='post', maxlen=350 )
		pred   = model.predict(encode)
		print( line )
		print( encode )
		print( pred[0])