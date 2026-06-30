import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path

from PIL import Image, ImageTk

from .app import Application


class MainWindow(tk.Tk):
    def __init__(self, application: Application):
        super().__init__()

        self.application = application
        self.item = None
        self.tk_image = None  # keep reference alive

        self.title("Segmentation Viewer")
        self.geometry("1000x800")

        # Top buttons
        top = tk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)

        tk.Button(
            top,
            text="Open annotations",
            command=self.open_annotations,
        ).pack(side="left", padx=4)

        tk.Button(
            top,
            text="Open segmentations",
            command=self.open_segmentations,
        ).pack(side="left", padx=4)

        tk.Button(
            top,
            text="Open images",
            command=self.open_images,
        ).pack(side="left", padx=4)

        tk.Button(
            top,
            text="Next image",
            command=self.next_image,
        ).pack(side="left", padx=4)

        # Segmentation selector
        selector_frame = tk.Frame(self)
        selector_frame.pack(fill="x", padx=8, pady=4)

        tk.Label(selector_frame, text="Segmentation mask:").pack(side="left")

        self.segmentation_var = tk.StringVar()
        self.segmentation_box = ttk.Combobox(
            selector_frame,
            textvariable=self.segmentation_var,
            state="readonly",
            width=20,
        )
        self.segmentation_box.pack(side="left", padx=8)
        self.segmentation_box.bind("<<ComboboxSelected>>", self.on_segmentation_changed)

        # Image display
        self.image_label = tk.Label(self, bg="black")
        self.image_label.pack(expand=True, fill="both", padx=8, pady=8)

        # Status
        self.status_label = tk.Label(self, text="Choose the three folders.")
        self.status_label.pack(fill="x", padx=8, pady=4)

    def open_annotations(self):
        path = filedialog.askdirectory(title="Choose annotations folder")
        if path:
            self.application.open_annotations(Path(path))
            self.try_load()

    def open_segmentations(self):
        path = filedialog.askdirectory(title="Choose segmentations folder")
        if path:
            self.application.open_segmentations(Path(path))
            self.try_load()

    def open_images(self):
        path = filedialog.askdirectory(title="Choose images folder")
        if path:
            self.application.open_images(Path(path))
            self.try_load()

    def next_image(self):
        if not self.application.loader:
            return

        self.application.next_item()
        self.item = self.application.current_item

        values = [
            f"Mask {i + 1}"
            for i in range(len(self.item.segmentations))
        ]

        self.segmentation_box["values"] = values

        if values:
            self.segmentation_box.current(0)
        else:
            self.segmentation_var.set("")

        self.status_label.config(
            text=f"Loaded: {self.item.image_path.name}"
        )

        self.display_current_image()

    def try_load(self):
        if not self.application.initialized:
            return

        self.application._try_initialize()

        if len(self.application.loader) == 0:
            self.status_label.config(text="No items found.")
            return

        self.application.current_index = 0
        self.item = self.application.current_item
        if not self.item:
            return

        values = [
            f"Mask {i + 1}"
            for i in range(len(self.item.segmentations))
        ]

        self.segmentation_box["values"] = values

        if values:
            self.segmentation_box.current(0)

        self.status_label.config(
            text=f"Loaded: {self.item.image_path.name}"
        )

        self.display_current_image()

    def on_segmentation_changed(self, event=None):
        self.display_current_image()

    def display_current_image(self):
        if self.item is None:
            return

        image = self.make_overlay_image()
        image = self.resize_for_display(image)

        self.tk_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.tk_image)

    def make_overlay_image(self) -> Image.Image:
        """
        Make original image grayscale and overlay selected mask in semi-transparent yellow.
        """
        if not self.item:
            raise RuntimeError("This should not happen.")
        base = self.item.image.convert("L").convert("RGBA")

        mask_index = self.segmentation_box.current()

        if mask_index < 0 or mask_index >= len(self.item.segmentations):
            return base.convert("RGB")

        mask = self.item.segmentations[mask_index].convert("L")

        yellow = Image.new(
            mode="RGBA",
            size=base.size,
            color=(255, 255, 0, 0),
        )

        alpha = mask.point(lambda p: 120 if p > 0 else 0)
        yellow.putalpha(alpha)

        result = Image.alpha_composite(base, yellow)
        return result.convert("RGB")

    def resize_for_display(self, image: Image.Image) -> Image.Image:
        image = image.copy()
        image.thumbnail((950, 700))
        return image


if __name__ == "__main__":
    app = Application()
    window = MainWindow(app)
    window.mainloop()