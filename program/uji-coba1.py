# Import Library
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load model CNN
model_path = 'X:/RAHMAN/fruit_classifier_model.h5'                                                                           
model = load_model(model_path)

# Inisialisasi label kelas sesuai dataset CNN
class_labels = ['Alpukat', 'Durian', 'Jeruk', 'Mangga', 'Pepaya']

# Inisialisasi GUI
def start_gui():
    root = tk.Tk()
    root.title("Sortasi Buah")

    # Fungsi untuk menangani gambar dari webcam
    def video_feed():
        # Tangkap frame dari webcam
        ret, frame = cap.read()
        if not ret:
            label.config(text="Gagal memuat kamera!")
            return

        # Proses frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (100, 100))  # Ukuran sesuai input model
        normalized_frame = frame_resized / 255.0  # Normalisasi

        # Prediksi kelas gambar
        predictions = model.predict(np.expand_dims(normalized_frame, axis=0))
        predicted_class = np.argmax(predictions)
        class_label = class_labels[predicted_class]
        accuracy = np.max(predictions) * 100

        # Tampilkan hasil prediksi pada GUI
        label.config(text=f"Jenis Buah: {class_label}")
        accuracy_label.config(text=f"Akurasi: {accuracy:.2f}%")

        # Tampilkan video pada GUI
        image = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=image)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Perbarui feed video setiap 10 milidetik
        video_label.after(10, video_feed)

    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Tidak dapat membuka webcam!")

    # Elemen GUI untuk menampilkan hasil
    label = ttk.Label(root, text="Jenis Buah: -", font=("Arial", 14))
    label.pack(pady=10)

    accuracy_label = ttk.Label(root, text="Akurasi: -", font=("Arial", 14))
    accuracy_label.pack(pady=10)

    video_label = ttk.Label(root)
    video_label.pack(pady=10)

    # Mulai video feed
    video_feed()

    # Jalankan GUI
    root.mainloop()

    # Setelah GUI ditutup, lepaskan kamera
    cap.release()

# Jalankan GUI
start_gui()
