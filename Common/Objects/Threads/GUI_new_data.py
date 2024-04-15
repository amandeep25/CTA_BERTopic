import tkinter as tk
from tkinter import filedialog

def send_data():
    upload_window = tk.Tk()
    upload_window.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Open file dialog for uploading data
    if file_path:
        print("Data uploaded successfully:", file_path)
    upload_window.destroy()

def receive_data():
    topics = ["Topic 1", "Topic 2", "Topic 3"]
    probabilities = [0.8, 0.6, 0.4]

    print("Topics:")
    for topic in topics:
        print(topic)

    print("\nProbabilities:")
    for prob in probabilities:
        print(prob)

# Create main window
root = tk.Tk()
root.title("Data Communication App")

# Send Data Button
send_button = tk.Button(root, text="Send Data", command=send_data)
send_button.pack()

# Receive Data Button
receive_button = tk.Button(root, text="Receive Data", command=receive_data)
receive_button.pack()

# Start the GUI event loop
root.mainloop()