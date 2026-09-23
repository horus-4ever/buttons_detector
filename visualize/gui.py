from pathlib import Path
from tkinter import ttk
import tkinter as tk
from .model import Graph, CurveToogle, CurveToogleSet, Application, FrameManager, GraphFrame
from .filedialog import FileDialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
import numpy as np


class CurveToggleView(ttk.Frame):
    def __init__(self, curve_toggle: CurveToogle, master = None):
        super().__init__(master=master)
        self.toggle = curve_toggle
        self._var = tk.BooleanVar(value=curve_toggle.visible)
        self._create_widgets()

    def _create_widgets(self):
        # define a common style
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("CurveToggleView.TCheckbutton", font=("Arial", 16))
        self.checkbox = ttk.Checkbutton(self, text=self.toggle.curve_name, variable=self._var, style="CurveToggleView.TCheckbutton")
        self.checkbox.pack()
        self._var.trace_add("write", self._on_change)

    def _on_change(self, a, b, c):
        self.toggle.toggle()


class CurveToggleSetView(ttk.Frame):
    def __init__(self, curves: CurveToogleSet, master = None):
        super().__init__(master=master)
        self.curves = curves
        self._create_widgets()

    def _create_widgets(self):
        self.toggle_views = []
        for curve_toggle in self.curves:
            curve_toggle_view = CurveToggleView(curve_toggle, master=self)
            curve_toggle_view.pack(side="top", anchor="w", pady=5)
            self.toggle_views.append(curve_toggle_view)


class GraphView(ttk.Frame):
    def __init__(self, graph: Graph, master = None):
        super().__init__(master=master)
        self.graph = graph
        self._create_widgets()
        self._create_graph()

    def _create_widgets(self):
        self.toggle_views = CurveToggleSetView(self.graph.toggles, master=self)
        self.toggle_views.pack(side="left", fill="x", padx=5)

    def _create_graph(self):
        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.axes = {}
        for curve_toggle in self.graph.toggles:
            curve_name = curve_toggle.curve_name
            curve_data = self.graph.data[curve_name]
            curve_x = np.arange(1, len(curve_data) + 1)
            new_ax, *_ = self.ax.plot(curve_x, curve_data, label=curve_name)
            self.axes[curve_name] = new_ax
            # link the toggle event
            curve_toggle.changed.add_listener(self._on_curve_toggle)
        # draw the graph for the tkinter integration
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

    def _on_curve_toggle(self, toggle: CurveToogle):
        self.axes[toggle.curve_name].set_visible(toggle.visible)
        self.canvas.draw_idle()


class GraphFrameView(ttk.Frame):
    def __init__(self, graph_frame: GraphFrame, master = None):
        super().__init__(master=master)
        self.graph_frame = graph_frame
        self.graph_frame.changed.add_listener(self._redraw)
        self._create_widgets()

    def _redraw(self, frame):
        for child in self.winfo_children():
            child.destroy()
        self._create_widgets()

    def _create_widgets(self):
        view = GraphView(self.graph_frame.graph, master=self)
        view.pack(fill="both", expand=True)


class FrameManagerView(ttk.Notebook):
    def __init__(self, frame_manager: FrameManager, master = None):
        super().__init__(master=master)
        self.frame_manager = frame_manager
        self._create_widgets()

    def _create_widgets(self):
        for name, frame in self.frame_manager:
            view = None
            if isinstance(frame, GraphFrame):
                view = GraphFrameView(frame, master=self)
            if view is None:
                raise ValueError(f"Frame is of type {type(frame)}, which is not a valid type.")
            self.add(view, text=name)


class MainMenu(tk.Menu):
    def __init__(self, application: Application, master=None):
        super().__init__(master=master)
        self.application = application
        self._create_widgets()

    def _create_widgets(self):
        self.add_command(label="Open", command=self._open_file)

    def _open_file(self):
        filename = FileDialog.open_file()
        if filename is not None:
            filename = Path(filename)
            self.application.open(filename)


class MainWindow(ttk.Frame):
    def __init__(self, application: Application, master = None):
        super().__init__(master=master)
        self.application = application
        self._create_widgets()

    def _create_widgets(self):
        # define the application menu
        self.menu = MainMenu(self.application, master=self)
        self.master.config(menu=self.menu) # type: ignore
        # define the widgets
        self.frame_manager_view = FrameManagerView(self.application.frame_manager)
        self.frame_manager_view.pack(fill="both", expand=True)
