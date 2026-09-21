import tkinter.ttk as ttk
import tkinter as tk
import tkinter.filedialog
from typing import Any, Callable

from dataset_tool.filedialog import FileDialog

from .data import GarmentEntry, GarmentInfo, GarmentSplit, Menu
from pathlib import Path
import json


class MenuBar(ttk.Frame):
    def __init__(self, menu: Menu, master=None):
        super().__init__(master)
        self.menu = menu
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.file_menu = tk.Menu(self.master)
        self.file_menu.add_command(label="Open Directory", command=self.open_directory)

    def open_directory(self):
        directory = FileDialog.select_directory()
        if directory:
            directory = Path(directory)
            self.menu.open_directory(directory)


class LabeledEntry(ttk.Frame):
    def __init__(self, label: str, widget_constructor: Callable, master=None):
        super().__init__(master)
        self.label = label
        self.widget_constructor = widget_constructor
        self.create_widgets()

    def create_widgets(self):
        self.label_widget = ttk.Label(self, text=self.label)
        self.label_widget.pack(anchor="n", side="left", padx=5)
        self.value_widget = self.widget_constructor(self)
        self.value_widget.pack(side="left", expand=True, fill="x", padx=5)


class GarmentInfoPopup(tk.Toplevel):
    def __init__(self, garment: GarmentEntry, master=None):
        super().__init__(master)
        self.garment_entry = garment
        self.garment = GarmentInfo(garment.garment)
        self.title(f"Garment Info: {self.garment_entry.name}")
        self.geometry("400x300")
        self.create_widgets()

    def create_widgets(self):
        info_frame = ttk.Frame(self)
        info_frame.pack(fill="both", expand=True, padx=10, pady=10)

        def _construct_textarea(master):
            text_widget = tk.Text(master, height=5)
            text_widget.insert("1.0", self.garment.description.get())
            return text_widget
        # add the garment information
        self.garment_type = LabeledEntry("Type", lambda master: ttk.Entry(master, textvariable=self.garment.type), master=info_frame)
        self.garment_type.pack(anchor="nw", fill="x", pady=2)
        self.garment_fastener = LabeledEntry("Fastener", lambda master: ttk.Entry(master, textvariable=self.garment.fastener), master=info_frame)
        self.garment_fastener.pack(anchor="nw", fill="x", pady=2)
        self.garment_assistive = LabeledEntry("Assistive", lambda master: ttk.Checkbutton(master, variable=self.garment.assistive), master=info_frame)
        self.garment_assistive.pack(anchor="nw", pady=2)
        self.garment_n_pairs = LabeledEntry("Number of Pairs", lambda master: ttk.Spinbox(master, from_=0, to=100, textvariable=self.garment.n_pairs), master=info_frame)
        self.garment_n_pairs.pack(anchor="nw", fill="x", pady=2)
        self.garment_description = LabeledEntry("Description", _construct_textarea, master=info_frame)
        self.garment_description.pack(anchor="nw", fill="x", pady=2)
        # now a separator
        separator = ttk.Separator(info_frame, orient="horizontal")
        separator.pack(fill="x", pady=10)
        # now a save button
        style = ttk.Style()
        style.configure("Save.TButton", foreground="white", background="#4CAF50")
        style.map("Save.TButton", background=[("active", "#45a049")])
        self.save_button = ttk.Button(info_frame, text="Save", command=self.save_garment_info, style="Save.TButton")
        self.save_button.pack(anchor="w", pady=5)

    def save_garment_info(self):
        description = self.garment_description.value_widget.get("1.0", tk.END).strip()
        self.garment.description.set(description)
        # set back to the garment variable
        self.garment.backpopulate()
        self.garment.save(self.garment_entry.path)
        # close the popup
        self.destroy()


class GarmentEntryView(ttk.Frame):
    def __init__(self, garment: GarmentEntry, garment_split: GarmentSplit, master=None):
        super().__init__(master)
        self.garment = garment
        self.garment_split = garment_split
        self.create_widgets()

    def create_widgets(self):
        self.label = ttk.Label(self, text=self.garment.name)
        self.label.pack(anchor="w", padx=5, pady=5)
        # contextual menu
        self.menu = tk.Menu(self.master, tearoff=False)
        self.menu.add_command(label="Train", command=lambda: self.move_to("train"))
        self.menu.add_command(label="Validation", command=lambda: self.move_to("val"))
        self.menu.add_command(label="Test", command=lambda: self.move_to("test"))
        # set the style
        entry_style = ttk.Style()
        entry_style.configure("GarmentEntry.TFrame", background="white")
        entry_style.configure("GarmentEntry.TLabel", background="white", foreground="black")
        entry_style.map("GarmentEntry.TLabel", background=[("active", "#e0e0ff")])
        entry_style.map("GarmentEntry.TFrame", background=[("active", "#e0e0ff")])
        self.label.configure(style="GarmentEntry.TLabel")
        self.configure(style="GarmentEntry.TFrame")
        # bind to event
        self.bind("<Button-3>", self._show_context_menu)
        self.bind("<Enter>", lambda e: self._set_active(True))
        self.bind("<Leave>", lambda e: self._set_active(False))
        self.bind("<Button-1>", lambda e: self._on_click())

    def _on_click(self):
        # shows the popup with the garment informations
        info_window = GarmentInfoPopup(self.garment, master=self)
        # set the popup to be modal
        info_window.transient(self.master) # type: ignore
        info_window.wait_visibility()
        info_window.grab_set()
        info_window.focus_set()
        self.master.wait_window(info_window)


    def _set_active(self, active: bool):
        if active:
            self.state(["active"])
            for widget in self.winfo_children():
                if isinstance(widget, ttk.Widget):
                    widget.state(["active"])
        else:
            self.state(["!active"])
            for widget in self.winfo_children():
                if isinstance(widget, ttk.Widget):
                    widget.state(["!active"])

    def _show_context_menu(self, event):
        self.menu.tk_popup(event.x_root, event.y_root)

    def move_to(self, split: str):
        self.garment_split.move_to(self.garment, self.garment.position, split)


class GarmentSplitView(ttk.Frame):
    def __init__(self, garment_split: GarmentSplit, master=None):
        super().__init__(master)
        self.garment_split = garment_split
        self.create_widgets()

    def redraw(self):
        # destroy all widgets and recreate them
        for widget in self.winfo_children():
            widget.destroy()
        self.create_widgets()

    def create_widgets(self):
        self.train_frame = ttk.LabelFrame(self, text="Train")
        self.train_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        for garment in self.garment_split.train:
            entry_view = GarmentEntryView(garment, self.garment_split, master=self.train_frame)
            entry_view.pack(side="top", fill="x")
            separator = ttk.Separator(self.train_frame, orient="horizontal")
            separator.pack(fill="x")

        self.val_frame = ttk.LabelFrame(self, text="Validation")
        self.val_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        for garment in self.garment_split.val:
            entry_view = GarmentEntryView(garment, self.garment_split, master=self.val_frame)
            entry_view.pack(side="top", fill="x")
            separator = ttk.Separator(self.val_frame, orient="horizontal")
            separator.pack(fill="x")

        self.test_frame = ttk.LabelFrame(self, text="Test")
        self.test_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        for garment in self.garment_split.test:
            entry_view = GarmentEntryView(garment, self.garment_split, master=self.test_frame)
            entry_view.pack(side="top", fill="x")
            separator = ttk.Separator(self.test_frame, orient="horizontal")
            separator.pack(fill="x")