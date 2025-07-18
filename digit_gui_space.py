import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import cv2
import os
import csv
from datetime import datetime

model = tf.keras.models.load_model("digit_cnn_model.h5")

log_file = "multi_digit_predictions.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Digit Index", "Predicted", "Timestamp"])

root = tk.Tk()
root.title("Multi-Digit Detector with Accuracy Log")
root.geometry("700x580")
root.configure(bg="#0b0f29")

canvas = tk.Canvas(root, width=680, height=400, bg='black', highlightbackground="#00ffe1")
canvas.pack(pady=20)

image = Image.new("L", (680, 400), 'black')
draw = ImageDraw.Draw(image)

def draw_lines(event):
    x, y = event.x, event.y
    canvas.create_oval(x-6, y-6, x+6, y+6, fill='white', outline='white')
    draw.ellipse([x-6, y-6, x+6, y+6], fill='white')

canvas.bind("<B1-Motion>", draw_lines)

def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, 680, 400], fill='black')
    result_label.config(text="Result: -")

def segment_and_predict_img(img_np, source="Canvas"):
    result_digits = ""
    _, thresh = cv2.threshold(img_np, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])

    with open(log_file, "a", newline="") as file:
        writer = csv.writer(file)
        for idx, (x, y, w, h) in enumerate(digit_boxes):
            if w > 5 and h > 10:
                digit_img = thresh[y:y+h, x:x+w]
                digit_img = cv2.copyMakeBorder(digit_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
                digit_img = cv2.resize(digit_img, (28, 28))
                digit_img = digit_img.reshape(1, 28, 28, 1).astype("float32") / 255.0
                pred = model.predict(digit_img, verbose=0)
                digit = np.argmax(pred)
                result_digits += str(digit)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([source, idx + 1, digit, timestamp])

    return result_digits if result_digits else "-"

def segment_and_predict():
    img = image.copy().resize((680, 400))
    img_np = np.array(img)
    result = segment_and_predict_img(img_np, source="Canvas")
    result_label.config(text=f"Result: {result}")

def camera_mode():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (680, 400))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        result = segment_and_predict_img(gray, source="Camera")
        cv2.putText(frame_resized, f"Result: {result}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Multi-Digit Detector", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==== Buttons ====
btn_frame = tk.Frame(root, bg="#0b0f29")
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="🔍 Predict", command=segment_and_predict, bg="#00ffe1", fg="#000",
          font=("Arial", 12, "bold"), width=12).grid(row=0, column=0, padx=10)

tk.Button(btn_frame, text="🧹 Clear", command=clear, bg="#ff0066", fg="#fff",
          font=("Arial", 12), width=12).grid(row=0, column=1, padx=10)

tk.Button(btn_frame, text="📷 Camera", command=camera_mode, bg="#1e90ff", fg="#fff",
          font=("Arial", 12), width=12).grid(row=0, column=2, padx=10)

result_label = tk.Label(root, text="Result: -", font=("Consolas", 18), fg="#00ff99", bg="#0b0f29")
result_label.pack(pady=20)

root.mainloop()
