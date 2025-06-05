import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Tentukan arsitektur model CNN
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Tentukan path ke direktori dataset
dataset_dir = 'X:\RAHMAN\dataset\durian\Train'

# Tentukan ukuran batch dan jumlah epoch
batch_size = 32
epochs = 10

# Preprocessing data menggunakan ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # validation_split menentukan proporsi data untuk validasi

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'  # menggunakan subset 'training' untuk data pelatihan
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'  # menggunakan subset 'validation' untuk data validasi
)

# Bangun model CNN
model = create_cnn_model((224, 224, 3), 3)  # Panggil fungsi create_cnn_model yang sudah dibuat sebelumnya

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Latih model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Simpan model setelah pelatihan
model.save('model_bawang_merah13.h5')
