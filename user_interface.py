import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np


from keras.models import load_model

# Load trained model
model = load_model('mymodel_imageclassifier.h5')

# Define your class labels
classes = {
    0: 'aeroplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

# Initialize the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Image Classification')
top.configure(background='#F2F2F2')

label = Label(top, background='#F2F2F2', font=('Arial', 15, 'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32, 32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    predictions = model.predict(image)
    pred = np.argmax(predictions)
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#2170EE', text=sign)


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image",
                        command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#2AD950', foreground='white',
                         font=('Arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
                            (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print(f"Error: {e}")


upload = Button(top, text="Upload an image", command=upload_image,
                padx=10, pady=5)

upload.configure(background='#364156', foreground='white',
                 font=('Arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Custom Image Classification",
                pady=20, font=('Arial', 20, 'bold'))

heading.configure(background='#F2F2F2', foreground='#2170EE')
heading.pack()
top.mainloop()
