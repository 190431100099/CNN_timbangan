import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set the path to the dataset folder
dataset_dir = 'X:/RAHMAN/dataset/'

# Image data generator untuk augmentasi gambar
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalisasi gambar
    rotation_range=30,         # Rotasi acak gambar
    width_shift_range=0.2,     # Pergeseran horizontal
    height_shift_range=0.2,    # Pergeseran vertikal
    shear_range=0.2,           # Shearing
    zoom_range=0.2,            # Zooming
    horizontal_flip=True,      # Pembalikan horizontal
    fill_mode='nearest'       # Mengisi area kosong setelah transformasi
)

# Membaca data training dari folder dan mengorganisasi ke dalam label
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),    # Ukuran gambar yang akan diproses
    batch_size=32,             # Jumlah gambar per batch
    class_mode='categorical'   # Untuk klasifikasi multi-kelas
)

# Membangun model CNN
model = Sequential()

# Layer konvolusi pertama
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer konvolusi kedua
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer konvolusi ketiga
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer untuk mengubah data 2D menjadi 1D
model.add(Flatten())

# Fully connected layer pertama
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout untuk mencegah overfitting

# Output layer dengan 5 kelas buah
model.add(Dense(5, activation='softmax'))

# Kompilasi model dengan optimizer Adam
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10
)

# Menyimpan model
model.save('fruit_classifier_model.h5')

print("Model training selesai dan disimpan.")

model_new=tf.keras.models.load_model('fruit_classifier_model.h5')
model_new.summery()

