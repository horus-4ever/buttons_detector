import os
import shutil
import subprocess
import tkinter as tk
from tkinter import filedialog


class FileDialog:
    @staticmethod
    def _desktop() -> str:
        return os.environ.get("XDG_CURRENT_DESKTOP", "").lower()

    @classmethod
    def _backend(cls) -> str:
        desktop = cls._desktop()
        if shutil.which("zenity"):
            return "zenity"
        return "tkinter"

    @staticmethod
    def _run(command: list[str]) -> str | None:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        str_result = result.stdout.strip() or None
        return str_result

    @classmethod
    def open_file(
        cls,
        *,
        title: str = "Open file",
        initialdir: str | None = None,
        parent: tk.Misc | None = None,
    ) -> str | None:

        backend = cls._backend()
        if backend == "zenity":
            command = [
                "zenity",
                "--file-selection",
                f"--title={title}",
            ]
            if initialdir:
                command.append(f"--filename={initialdir}/")
            return cls._run(command)
        
        result = filedialog.askopenfilename(
            parent=parent,
            title=title,
            initialdir=initialdir,
        )

        return result or None

    @classmethod
    def save_file(
        cls,
        *,
        title: str = "Save file",
        initialdir: str | None = None,
        filename: str | None = None,
        parent: tk.Misc | None = None,
    ) -> str | None:

        backend = cls._backend()
        if backend == "zenity":
            command = [
                "zenity",
                "--file-selection",
                "--save",
                "--confirm-overwrite",
                f"--title={title}",
            ]
            if filename:
                path = (
                    os.path.join(initialdir, filename)
                    if initialdir
                    else filename
                )
                command.append(f"--filename={path}")
            return cls._run(command)
        result = filedialog.asksaveasfilename(
            parent=parent,
            title=title,
            initialdir=initialdir,
            initialfile=filename,
        )
        return result or None

    @classmethod
    def select_directory(
        cls,
        *,
        title: str = "Select directory",
        initialdir: str | None = None,
        parent: tk.Misc | None = None,
    ) -> str | None:

        backend = cls._backend()
        if backend == "zenity":
            command = [
                "zenity",
                "--file-selection",
                "--directory",
                f"--title={title}",
            ]
            if initialdir:
                command.append(f"--filename={initialdir}/")
            return cls._run(command)

        result = filedialog.askdirectory(
            parent=parent,
            title=title,
            initialdir=initialdir,
        )
        return result or None