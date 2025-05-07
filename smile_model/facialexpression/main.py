# To load images and do preprocessing and data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

# Dropout and Flatten are used to prevent overfitting
# Dropout is also helpful in classification layers

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# Conv2D is for convolution operations, MaxPooling2D for downsampling

# os is used to import and count images from the directory 
import os

# Set training and validation directory paths
train_data_dir = 'data/train'
validation_data_dir = 'data/test'

# Data augmentation is required since there are only 7 classes
# We need varied training data to increase accuracy
train_datagen = ImageDataGenerator(
    rescale=1./255, # Rescale pixel values to be between 0 and 1
    shear_range=0.2, # Shear the image by 20%
    zoom_range=0.2, # Zoom into the image by 20%
    horizontal_flip=True, # Flip the image horizontally
    fill_mode='nearest' # Fill in new pixels created by transformations
)

# For validation, we only rescale the pixel values
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess training data from the directory
train_generator = train_datagen.flow_from_directory(
    train_data_dir, # Directory of the training data
    color_mode='grayscale', # Convert images to grayscale
    target_size=(48, 48), # Resize the images to 48x48 pixels
    batch_size=32, # Batch size of 32 images
    class_mode='categorical', # Categorical since it's a multi-class classification
    shuffle=True # Shuffle the images to avoid bias
)

# Load and preprocess validation data from the directory
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir, # Directory of the validation data
    color_mode='grayscale', # Convert images to grayscale
    target_size=(48, 48), # Resize the images to 48x48 pixels
    batch_size=32, # Batch size of 32 images
    class_mode='categorical', # Categorical since it's a multi-class classification
    shuffle=False # Do not shuffle validation data for consistent evaluation
)

# Define class labels (for reference or plotting later)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Fetch a batch of images and labels just to verify loading
img, label = next(train_generator)

# Create a sequential model
model = Sequential()

# Add layers to the model
# Input convolutional layer
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))

# Second convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # Max pooling to reduce spatial dimensions
model.add(Dropout(0.25)) # Dropout layer to prevent overfitting

# Third convolutional block
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fourth convolutional block
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten the output to feed into dense layers
model.add(Flatten())
model.add(Dense(512, activation='relu')) # Fully connected dense layer with ReLU
model.add(Dropout(0.5)) # Dropout with 50% rate for strong regularization

# Output layer with 7 classes using softmax activation
model.add(Dense(7, activation='softmax'))

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
print(model.summary())

# Count number of training images
num_train_imgs = 0
for root, dirs, files in os.walk(train_data_dir):
    num_train_imgs += len(files)

# Count number of validation images
num_test_imgs = 0
for root, dirs, files in os.walk(validation_data_dir):
    num_test_imgs += len(files)

# Define number of epochs for training
epochs = 30

# Train the model using the generator
history = model.fit(
    train_generator,
    steps_per_epoch=num_train_imgs // 32, # Number of steps per epoch
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_test_imgs // 32 # Number of validation steps
)

# Save the trained model to a file
model.save('model_file.h5')
