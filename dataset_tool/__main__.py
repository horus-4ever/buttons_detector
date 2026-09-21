from pathlib import Path
import tkinter.ttk as ttk
import tkinter as tk
import tkinter.filedialog
from .gui import GarmentEntryView, GarmentSplitView, MenuBar
from .data import Application, GarmentEntry, GarmentSplit
from .filedialog import FileDialog
import json


class MainWindow(ttk.Frame):
    def __init__(self, application: Application, master=None):
        super().__init__(master)
        self.application = application
        self.pack(fill="both", expand=True)
        self.create_widgets()

    def create_widgets(self):
        self.menu_bar = MenuBar(menu=self.application.menu, master=self)
        self.menu_bar.pack(side="top", fill="x")
        self.master.configure(menu=self.menu_bar.file_menu) # type: ignore
        self.garment_split_view = GarmentSplitView(self.application.garment_split, master=self)
        self.garment_split_view.pack(side="top", fill="both", expand=True)
        # add a separator
        separator = ttk.Separator(self, orient="horizontal")
        separator.pack(side="top", fill="x", pady=5)
        # now an export button
        style = ttk.Style()
        style.configure("Export.TButton", font=("Arial", 12), background="#a0e0ff", padding=5)
        style.map("Export.TButton", background=[("active", "#80c0ff")])
        self.export_button = ttk.Button(self, text="Export Garment Split", command=self.export_garment_split, style="Export.TButton")
        self.export_button.pack(side="top", pady=5)
        # link events
        self.application.garment_split.changed_event.add_listener(lambda *_, **__: self.garment_split_view.redraw())

    def export_garment_split(self):
        # export the garment split to a json file
        export_path = FileDialog.save_file()
        if not export_path:
            return
        export_path = Path(export_path)
        filename = export_path.name[:len(export_path.name) - len("".join(export_path.suffixes))]
        export_path = export_path.parent / f"{filename}.dataset.json"
        export_data = self.application.garment_split.to_json()
        with open(export_path, "w") as file:
            json.dump(export_data, file, indent=4)
        


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Dataset Tool")
    root.geometry("800x600")
    app = Application()
    main_window = MainWindow(application=app, master=root)
    main_window.mainloop()
