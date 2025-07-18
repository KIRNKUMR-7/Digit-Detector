import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import cv2
import csv
import os
import tensorflow as tf
from datetime import datetime

# Load the CNN model
model = tf.keras.models.load_model("digit_cnn_model.h5")

# ==== CSV Logging ====
csv_file = "prediction_log.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Digit", "Confidence (%)", "Timestamp"])

# ==== Animated Button ====
class AnimatedButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        self.default_bg = kwargs.get("bg", "#0b0f29")
        self.hover_bg = kwargs.get("activebackground", "#00ffe1")
        self.default_fg = kwargs.get("fg", "#00ffe1")
        self.hover_fg = kwargs.get("activeforeground", "#0b0f29")

        super().__init__(master, **kwargs)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.config(bg=self.hover_bg, fg=self.hover_fg, relief="sunken", font=("Orbitron", 12, "bold"))

    def on_leave(self, e):
        self.config(bg=self.default_bg, fg=self.default_fg, relief="raised", font=("Orbitron", 10))

# ==== GUI Window ====
root = tk.Tk()
root.title("Digit Detector")
root.configure(bg="#0b0f29")

canvas = tk.Canvas(root, width=280, height=280, bg='black', cursor="cross", bd=0, highlightthickness=0)
canvas.pack(pady=10)

image = Image.new("L", (280, 280), 'black')
draw = ImageDraw.Draw(image)

def draw_lines(event):
    x, y = event.x, event.y
    canvas.create_oval(x-8, y-8, x+8, y+8, fill='white', outline='white')
    draw.ellipse([x-8, y-8, x+8, y+8], fill='white')

canvas.bind("<B1-Motion>", draw_lines)

# ==== Prediction from Canvas ====
def predict():
    img = image.resize((28, 28))
    img = ImageOps.invert(img)
    img = np.array(img).reshape(1, 28, 28, 1).astype("float32") / 255.0
    pred = model.predict(img)
    digit = np.argmax(pred)
    confidence = np.max(pred) * 100
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Show result
    messagebox.showinfo("Prediction", f"Digit: {digit}\nConfidence: {confidence:.2f}%\n\nGemini:\nThis prediction was made using a CNN trained on MNIST dataset with 99%+ accuracy.")
    
    # Save to CSV
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Canvas", digit, f"{confidence:.2f}", timestamp])

    print(f"[Canvas] Digit: {digit} | Confidence: {confidence:.2f}% | {timestamp}")

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, 280, 280], fill='black')

# ==== Camera Input + Prediction ====
def open_camera_window():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(gray, (28, 28))
        roi = cv2.bitwise_not(roi)
        input_img = roi.reshape(1, 28, 28, 1).astype("float32") / 255.0
        pred = model.predict(input_img)
        digit = np.argmax(pred)
        confidence = np.max(pred) * 100
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Show on screen
        cv2.putText(frame, f"Digit: {digit} ({confidence:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Live Digit Detector", frame)

        # Save to CSV
        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Camera", digit, f"{confidence:.2f}", timestamp])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==== Buttons ====
frame = tk.Frame(root, bg="#0b0f29")
frame.pack(pady=10)

btn_style = {
    "bg": "#0b0f29",
    "fg": "#00ffe1",
    "activebackground": "#00ffe1",
    "activeforeground": "#0b0f29",
    "font": ("Orbitron", 10),
    "width": 12,
    "height": 1,
    "borderwidth": 2,
    "relief": "raised"
}

AnimatedButton(frame, text="Predict", command=predict, **btn_style).grid(row=0, column=0, padx=10)
AnimatedButton(frame, text="Clear", command=clear_canvas, **btn_style).grid(row=0, column=1, padx=10)
AnimatedButton(frame, text="Camera", command=open_camera_window, **btn_style).grid(row=0, column=2, padx=10)

# ==== Title ====
tk.Label(root, text="DIGIT DETECTOR", font=("Orbitron", 18, "bold"),
         bg="#0b0f29", fg="#00ffe1").pack(pady=10)

root.mainloop()
