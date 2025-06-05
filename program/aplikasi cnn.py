import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import serial
import time

# Load model
model = load_model('model_bawang_merah15.h5')

# Inisialisasi label kelas
class_labels = ['Kelas 1', 'Kelas 2', 'Kelas Super', 'Kosong']

# Inisialisasi GUI
root = tk.Tk()
root.title("Deteksi Objek Realtime")

# Inisialisasi koneksi serial
serial_port = serial.Serial('COM7', 9600)  # Ganti 'COM3' dengan port yang sesuai

# Variabel untuk menyimpan jumlah bawang merah yang terdeteksi setiap kelas
count = {'Kelas 1': 0, 'Kelas 2': 0, 'Kelas Super': 0, 'Kosong': 0}

# Fungsi untuk menangani gambar dari webcam
def video_feed():
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 480))
    
    # Proses prediksi gambar
    processed_frame = cv2.resize(frame, (224, 224))
    processed_frame = processed_frame / 255.0  # Normalisasi
    
    # Prediksi kelas gambar
    predictions = model.predict(np.expand_dims(processed_frame, axis=0))
    predicted_class = np.argmax(predictions)
    class_label = class_labels[predicted_class]

    # Tampilkan hasil prediksi pada GUI
    label.config(text="Kelas: " + class_label)
    
    # Kirim hasil prediksi ke Arduino melalui serial
    serial_port.write(str(predicted_class).encode())
    time.sleep(0.1)  # Tunggu sebentar untuk memastikan Arduino menerima pesan
    
    # Update count bawang merah yang terdeteksi
    count[class_label] += 1
    count_label.config(text="Jumlah Bawang Merah: " + str(count))
    
    # Ubah gambar menjadi format yang sesuai untuk Tkinter
    image = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=image)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    
    # Perbarui GUI setiap 10 milidetik
    video_label.after(10, video_feed)

# Fungsi untuk memulai atau menghentikan motor DC
def toggle_motor():
    if motor_button.config('text')[-1] == 'Start':
        motor_button.config(text='Stop')
        # Kirim sinyal untuk memulai motor ke Arduino melalui serial
        serial_port.write(b'start')
    else:
        motor_button.config(text='Start')
        # Kirim sinyal untuk menghentikan motor ke Arduino melalui serial
        serial_port.write(b'stop')

# Fungsi untuk mengatur kecepatan motor DC
def set_speed(speed):
    # Kirim nilai kecepatan ke Arduino melalui serial
    serial_port.write(str(speed).encode())

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Tidak dapat membuka webcam")

# Buat label untuk menampilkan hasil kelas
label = ttk.Label(root, text="")
label.pack()

# Buat label untuk menampilkan video dari webcam
video_label = ttk.Label(root)
video_label.pack()

# Buat tombol untuk memulai/menghentikan motor
motor_button = ttk.Button(root, text="Start", command=toggle_motor)
motor_button.pack()

# Buat slider untuk mengatur kecepatan motor
speed_slider = ttk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, command=set_speed)
speed_slider.set(128)  # Kecepatan default
speed_slider.pack()

# Buat label untuk menampilkan jumlah bawang merah yang terdeteksi
count_label = ttk.Label(root, text="Jumlah Bawang Merah: " + str(count))
count_label.pack()

# Mulai feed video
video_feed()

# Jalankan GUI
root.mainloop()

# Setelah GUI ditutup, lepaskan kamera dan tutup port serial
cap.release()
serial_port.close()
