from .gui import MainWindow
from .model import Application
import tkinter as tk
import torch


if __name__ == "__main__":
    # get the torch device
    device = torch.device("cpu")
    # create the application
    root = tk.Tk()
    root.title("Dataset Tool")
    root.geometry("800x600")
    application = Application(device=device)
    main_window = MainWindow(application, master=root)
    main_window.pack(fill="both", expand=True)
    main_window.mainloop()