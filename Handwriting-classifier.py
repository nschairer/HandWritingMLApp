import keras
from keras.datasets import mnist
from keras.models import Sequential
#can add as many layers to Sequential model, lets you build your down
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten#convolution layer, pooling layer,Fully connected layer,

#Preparing the data
pixel_width = 28
pixel_height = 28
num_of_classes = 10
batch_size = 32#same as convoluted layers
epochs = 10
#labels is what the image is categorized as, features are what it is broken down into

(features_train, labels_train), (features_test, labels_test) = mnist.load_data()

# the 1 is the pixel depth
features_train = features_train.reshape(features_train.shape[0], pixel_width, pixel_height ,1)
features_test = features_test.reshape(features_test.shape[0], pixel_width, pixel_height ,1)

input_shape = (pixel_width, pixel_height, 1)

features_train = features_train.astype('float32')
features_test = features_test.astype('float32')

#divides value by 255 then sets equal to that value, makes it % of RGB
features_train /= 255
features_test /= 255


#flatten label data into binary matrix [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.] = 2
#this makes it easier to interact with the prediciton values
labels_train = keras.utils.to_categorical(labels_train, num_of_classes)
labels_test = keras.utils.to_categorical(labels_test, num_of_classes)

#Building/Visualizing the CNN

model = Sequential()
#32 filters, kernalsize 3,3 the pixel grid for each filter, ReLU to turn negatives to 0,
#convolution shaves off outside pixels lowering size to 26,26
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
print("POST CONV2D: ", model.output_shape)
#pool size = 2 by 2, strides default to pool size unless specified
model.add(MaxPooling2D(pool_size=(2,2)))
print("POST MAXPOOLING2D: ", model.output_shape)
#Droput layer randomizes the data so every instance isn't the same, % rate for values to equal zero, avoids overfitting data aka one way to do something, gives flex
model.add(Dropout(0.25))#25% are dropped
print("POST DROPOUT: ", model.output_shape)

#flatten model into one stack of dataset, makes 5408 nodes
model.add(Flatten())
print("POST FLATTEN: ", model.output_shape)

#need to reduce to 128 nodes
model.add(Dense(128, activation='relu'))
print('POST DENSE: ', model.output_shape)

#ouput layer 10 nodes, softmax converts arrays of values into array of 10 values between 0-1, highest value is the prediction
model.add(Dense(num_of_classes, activation='softmax'))
print('POST DENSE2: ', model.output_shape)

#need to compile model
#loss calculates the models loss, categorical cross entropy calculates accuracy between train and test
#Adadelta is our optimizer, specialized for photos
#categorical_crossentropy uses categorical accuracy
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
#batch_size is number of guesses before changing the line, epochs is number of iterations through training data, verbose prints, validation data is test data
#features and labels are (x,y)
model.fit(features_train, labels_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(features_test,labels_test))

#get model accuracy
score = model.evaluate(features_test, labels_test, verbose=0)

model.save('/Users/NoahSchairer/Desktop/handwriting_model.h5')

import coremltools
coreml_model = coremltools.converters.keras.convert(model, input_names=['image'], image_input_names='image')

coreml_model.author = 'Noah Schairer'
coreml_model.license = 'MIT'
coreml_model.short_description = 'Predicts the handwritten character passed in as a number between 1-9.'
coreml_model.input_description['image'] = 'A 28x28 pixel grayscale image.'
coreml_model.output_description['output1'] = 'A Multiarray where the index with the greatest float value (0-1) is the recognized digit.'

coreml_model.save('/Users/NoahSchairer/Desktop/handwriting.mlmodel')
