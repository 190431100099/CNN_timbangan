import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import serial
import time

# Load model
model = load_model('X:/RAHMAN/fruit_classifier_model_baru.h5')

# Inisialisasi label kelas
class_labels = ['alpukat', 'apel', 'durian', 'jeruk']

# Inisialisasi GUI
root = tk.Tk()
root.title("Sortasi buah")

# Inisialisasi koneksi serial
serial_port = serial.Serial('COM3', 9600)  # Ganti 'COM3' dengan port yang sesuai
serial_port.timeout = 1

# Variabel untuk melacak waktu pengiriman nama buah
last_sent_time = time.time()
send_interval = 10.3  # Interval pengiriman data dalam detik

# Fungsi untuk menangani gambar dari webcam
def video_feed():
    global last_predicted_class, last_sent_time
    
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (1280, 720))
    
    # Proses prediksi gambar
    processed_frame = cv2.resize(frame, (100, 100))
    processed_frame = processed_frame / 255.0  # Normalisasi
    
    # Prediksi kelas gambar
    predictions = model.predict(np.expand_dims(processed_frame, axis=0))
    predicted_class = np.argmax(predictions)
    class_label = class_labels[predicted_class]
    accuracy = np.max(predictions) * 100  # Mengambil nilai akurasi maksimum dan mengonversi menjadi persentase
    
    # Tampilkan hasil prediksi pada GUI
    label.config(text="Kelas: " + class_label)
    accuracy_label.config(text="Akurasi: {:.2f}%".format(accuracy))  # Menampilkan akurasi prediksi

    # Kirim hasil prediksi ke Arduino melalui serial
    current_time = time.time()
    if current_time - last_sent_time >= send_interval:
        serial_port.write((class_label + '\n').encode())  # Menambahkan '\n' untuk memudahkan pembacaan di Arduino
        last_sent_time = current_time  # Update waktu terakhir pengiriman
    
    # Ubah gambar menjadi format yang sesuai untuk Tkinter
    image = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=image)  
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    
    # Perbarui GUI setiap 10 milidetik
    video_label.after(10, video_feed)

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Tidak dapat membuka webcam") 

# Buat label untuk menampilkan hasil kelas
label = ttk.Label(root, text="", font=("Arial", 12))
label.pack()

# Buat label untuk menampilkan akurasi prediksi
accuracy_label = ttk.Label(root, text="", font=("Arial", 12))
accuracy_label.pack()

# Buat label untuk menampilkan video dari webcam
video_label = ttk.Label(root)
video_label.pack()

# Mulai feed video
video_feed()

# Jalankan GUI
root.mainloop()

# Setelah GUI ditutup, lepaskan kamera dan tutup port serial
cap.release()
serial_port.close()
