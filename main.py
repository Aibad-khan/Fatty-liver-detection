from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np

# Define the custom augmentation function
def custom_augment(image):
    # Example augmentation: Rotate, shift, zoom, etc.
    # Implement your custom augmentations here for future
    return image  # Placeholder for custom augmentation logic

# Apply the custom augmentation on-the-fly during training
def custom_augmentation_generator(generator):
    while True:
        batch_x, batch_y = next(generator)
        augmented_batch_x = np.array([custom_augment(x) for x in batch_x])
        yield augmented_batch_x, batch_y

# Base directory containing 'fatty' and 'normal' folders
train_dir = r"your directory path"

# Image data generator for augmenting the images
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3,  # 30% for validation
    rotation_range=20,      # Small rotation to vary orientation (±20 degrees)
    width_shift_range=0.15,  # Shift image width by ±15%
    height_shift_range=0.15, # Shift image height by ±15%
    brightness_range=[0.8, 1.2],  # Random brightness adjustment
    zoom_range=0.1,         # Random zoom in/out by ±10%
    horizontal_flip=True,   # Flip images horizontally
    fill_mode='nearest'
)

# Load training and validation sets
original_train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  
    batch_size=32,
    class_mode='binary',
    subset='training'
)

original_validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Wrap the original generators with the custom augmentation
train_generator = custom_augmentation_generator(original_train_generator)
validation_generator = custom_augmentation_generator(original_validation_generator)

# Get the number of steps per epoch for training and validation
steps_per_epoch = len(original_train_generator)
validation_steps = len(original_validation_generator)

# Load the VGG16 model without the top fully connected layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze the last few layers
for layer in base_model.layers[:-6]:
    layer.trainable = False
for layer in base_model.layers[-6:]:
    layer.trainable = True

# Add custom layers on top
x = Flatten()(base_model.output)

# First Dense layer with Batch Normalization
x = Dense(128, activation='relu', kernel_regularizer=l2(0.02))(x)
x = BatchNormalization()(x)  # Add Batch Normalization after Dense layer
x = Dropout(0.3)(x)  # Dropout layer with 30% dropout rate

# Output layer
x = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.02))(x)

# Create the new model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping and learning rate scheduler callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',       # Monitor validation loss
    patience=5,               # Stop if no improvement for 5 epochs
    restore_best_weights=True # Restore model weights from the epoch with the best validation loss
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',       # Monitor validation loss
    factor=0.5,               # Reduce learning rate by a factor of 0.5
    patience=3,               # Reduce if no improvement for 3 epochs
    min_lr=1e-6               # Lower bound for learning rate
)

# Train the model
history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr]
)
