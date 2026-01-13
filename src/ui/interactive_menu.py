"""
Interactive menu for video and method selection.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional, Tuple


class VideoMethodSelector:
    """
    Interactive GUI for selecting:
    1. Input video from data/videos/input/
    2. Localization method (Task 1, Task 2, PnP)
    """
    
    def __init__(self, videos_dir: str = "data/videos/input"):
        self.videos_dir = Path(videos_dir)
        self.selected_video: Optional[str] = None
        self.selected_method: Optional[str] = None
        
        # Method mapping
        self.methods = {
            "Task 1: Omografia (4 punti targa)": "homography",
            "Task 2: Punto di Fuga (luci notturne)": "vanishing_point",
            "PnP: Metodo Diretto (confronto)": "pnp"
        }
        
        self.root = None
        self.video_listbox = None
        self.method_var = None
        
    def get_available_videos(self):
        """Scan videos directory for .mp4 files."""
        if not self.videos_dir.exists():
            return []
        
        videos = list(self.videos_dir.glob("*.mp4"))
        return sorted([v.name for v in videos])
    
    def create_gui(self):
        """Create the selection GUI."""
        self.root = tk.Tk()
        self.root.title("Vehicle 3D Localization - Setup")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Title
        title_label = tk.Label(
            self.root,
            text="🚗 Vehicle 3D Localization System",
            font=("Arial", 16, "bold"),
            pady=20
        )
        title_label.pack()
        
        # --- Video Selection Section ---
        video_frame = ttk.LabelFrame(
            self.root,
            text="1. Seleziona Video",
            padding=20
        )
        video_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Video list
        video_scroll = ttk.Scrollbar(video_frame)
        video_scroll.pack(side="right", fill="y")
        
        self.video_listbox = tk.Listbox(
            video_frame,
            height=6,
            yscrollcommand=video_scroll.set,
            font=("Courier", 10)
        )
        self.video_listbox.pack(fill="both", expand=True)
        video_scroll.config(command=self.video_listbox.yview)
        
        # Populate video list
        videos = self.get_available_videos()
        if not videos:
            self.video_listbox.insert(0, "⚠️ Nessun video trovato in data/videos/input/")
            self.video_listbox.config(state="disabled")
        else:
            for video in videos:
                self.video_listbox.insert(tk.END, video)
        
        # --- Method Selection Section ---
        method_frame = ttk.LabelFrame(
            self.root,
            text="2. Seleziona Metodo di Localizzazione",
            padding=20
        )
        method_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.method_var = tk.StringVar(value="Task 2: Punto di Fuga (luci notturne)")
        
        for method_name in self.methods.keys():
            rb = ttk.Radiobutton(
                method_frame,
                text=method_name,
                variable=self.method_var,
                value=method_name
            )
            rb.pack(anchor="w", pady=5)
        
        # Method descriptions
        desc_text = tk.Text(method_frame, height=4, wrap="word", font=("Arial", 9))
        desc_text.pack(fill="x", pady=10)
        desc_text.insert("1.0", 
            "Task 1: Ambiente diurno, targa visibile (omografia 4 punti)\n"
            "Task 2: Ambiente notturno, solo luci posteriori (punto di fuga)\n"
            "PnP: Metodo alternativo per confronto (solvePnP diretto)"
        )
        desc_text.config(state="disabled")
        
        # --- Action Buttons ---
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        start_btn = ttk.Button(
            button_frame,
            text="▶ Avvia Processing",
            command=self.on_start,
            width=20
        )
        start_btn.pack(side="left", padx=10)
        
        cancel_btn = ttk.Button(
            button_frame,
            text="✖ Annulla",
            command=self.on_cancel,
            width=20
        )
        cancel_btn.pack(side="left", padx=10)
        
    def on_start(self):
        """Handle start button click."""
        # Get selected video
        selection = self.video_listbox.curselection()
        if not selection:
            messagebox.showerror(
                "Errore",
                "Seleziona un video dalla lista!"
            )
            return
        
        self.selected_video = self.video_listbox.get(selection[0])
        
        # Check if video exists (not the warning message)
        if self.selected_video.startswith("⚠️"):
            messagebox.showerror(
                "Errore",
                "Nessun video disponibile. Aggiungi file .mp4 in data/videos/input/"
            )
            return
        
        # Get selected method
        method_name = self.method_var.get()
        self.selected_method = self.methods[method_name]
        
        # Confirm selection
        confirm = messagebox.askyesno(
            "Conferma",
            f"Video: {self.selected_video}\n"
            f"Metodo: {method_name}\n\n"
            f"Procedere con il processing?"
        )
        
        if confirm:
            self.root.quit()
            self.root.destroy()
    
    def on_cancel(self):
        """Handle cancel button click."""
        self.selected_video = None
        self.selected_method = None
        self.root.quit()
        self.root.destroy()
    
    def run(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Run the interactive menu.
        
        Returns:
            (video_filename, method_name) or (None, None) if cancelled
        """
        self.create_gui()
        self.root.mainloop()
        
        return self.selected_video, self.selected_method


def select_video_and_method() -> Tuple[Optional[str], Optional[str]]:
    """
    Convenience function to run the selector.
    
    Returns:
        (video_path, method) tuple
        - video_path: full path to selected video or None
        - method: 'homography' | 'vanishing_point' | 'pnp' or None
    """
    selector = VideoMethodSelector()
    video_name, method = selector.run()
    
    if video_name and method:
        video_path = Path("data/videos/input") / video_name
        return str(video_path), method
    
    return None, None


if __name__ == "__main__":
    # Test the menu
    video, method = select_video_and_method()
    
    if video and method:
        print(f"✓ Video selezionato: {video}")
        print(f"✓ Metodo selezionato: {method}")
    else:
        print("✗ Selezione annullata")