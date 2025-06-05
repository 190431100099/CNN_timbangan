import tensorflow as tf
from tensorflow.keras import layers, models

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

# Tentukan ukuran input gambar dan jumlah kelas
input_shape = (224, 224, 3)  # Ukuran gambar input
num_classes = 3  # 3 kelas: super, kelas 1, dan kelas 2

# Bangun model CNN
model = create_cnn_model(input_shape, num_classes)

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Tampilkan ringkasan model
model.summary()
