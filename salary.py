import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import Tk, Label, messagebox
from PIL import Image, ImageTk
import itertools

# Correct file path using raw string
csv_file_path = r"D:\Codes\projects\NIELIT mini project\salary_data.csv"

# Load and prepare the dataset
df = pd.read_csv(csv_file_path)

# Rest of your code...

# Splitting the dataset into training and testing sets
X = df[["YearsExperience"]]
y = df["Salary"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Testing the model
y_pred = model.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("R2 score: %.2f" % r2_score(y_test, y_pred))


# Define functions
def predict_salary():
    try:
        years_experience = float(entry.get())
        prediction = model.predict(np.array([[years_experience]]))[0]
        messagebox.showinfo(
            "Predicted Salary", f"The predicted salary is: RS: {prediction:.2f}"
        )
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter a valid number")


def animate_label():
    current_color = next(color_cycle)
    title_label.config(fg=current_color)
    root.after(500, animate_label)


# Creating the GUI
root = Tk()
root.title("Salary Prediction")

# Styling
style = ttk.Style()
style.configure("TLabel", font=("Helvetica", 12))
style.configure("TEntry", font=("Helvetica", 12))
style.configure("TButton", font=("Helvetica", 12))

# Color cycle for animation
color_cycle = itertools.cycle(["red", "orange", "yellow", "green", "blue", "purple"])

# Creating Widgets
title_label = Label(
    root,
    text="Salary Prediction Based on Years of Experience",
    font=("Helvetica", 16, "bold"),
)
title_label.pack(pady=20)

frame = ttk.Frame(root)
frame.pack(pady=10)

years_label = ttk.Label(frame, text="Years of Experience:")
years_label.grid(row=0, column=0, padx=10, pady=10)

entry = ttk.Entry(frame, width=20)
entry.grid(row=0, column=1, padx=10, pady=10)

predict_button = ttk.Button(
    root, text="Predict", command=predict_salary, bootstyle=SUCCESS
)
predict_button.pack(pady=20)

# Animation
animate_label()

# Adding an Image
image_path = (
    r"D:\Codes\projects\NIELIT mini project\ai.png"  # Use raw string for the path
)

try:
    image = Image.open(image_path)
    photo = ImageTk.PhotoImage(image)

    # Create a Label to display the image
    image_label = Label(root, image=photo)
    image_label.pack(pady=20)

    # Store the image reference in the root window to prevent garbage collection
    root.image = photo
except FileNotFoundError:
    messagebox.showerror("Error", f"File not found: {image_path}")

# Run the GUI
root.mainloop()
