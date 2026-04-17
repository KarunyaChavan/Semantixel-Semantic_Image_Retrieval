import tkinter as tk
import ttkbootstrap as ttk
from settings.tooltip import CreateToolTip


class GoogleDriveManager:
    def __init__(self, parent, config):
        self.parent = parent
        self.config = config
        self.google_drive_config = self.config.setdefault("google_drive", {})
        self.setup_ui()

    def setup_ui(self):
        drive_frame = ttk.LabelFrame(self.parent, text="Google Drive")
        drive_frame.pack(fill=tk.X, padx=10, pady=10)

        self.enabled_var = tk.BooleanVar(value=self.google_drive_config.get("enabled", False))
        enabled_frame = ttk.Frame(drive_frame)
        enabled_frame.pack(fill=tk.X, pady=5)
        enabled_label = ttk.Label(enabled_frame, text="Enable Google Drive:")
        enabled_label.pack(side=tk.LEFT, padx=5)
        CreateToolTip(enabled_label, "Enable indexing and serving images from Google Drive.")
        tk.Checkbutton(enabled_frame, variable=self.enabled_var).pack(side=tk.LEFT, padx=5)

        self.client_secret_var = tk.StringVar(value=self.google_drive_config.get("client_secret_file", ""))
        self.token_file_var = tk.StringVar(value=self.google_drive_config.get("token_file", "google_drive_token.json"))
        self.redirect_uri_var = tk.StringVar(value=self.google_drive_config.get("redirect_uri", ""))
        self.folder_ids_var = tk.StringVar(value=", ".join(self.google_drive_config.get("folder_ids", [])))
        self.include_shared_drives_var = tk.BooleanVar(value=self.google_drive_config.get("include_shared_drives", False))

        self._add_text_field(drive_frame, "Client Secret File:", self.client_secret_var)
        self._add_text_field(drive_frame, "Token File:", self.token_file_var)
        self._add_text_field(drive_frame, "Redirect URI:", self.redirect_uri_var)
        self._add_text_field(drive_frame, "Folder IDs (comma-separated):", self.folder_ids_var)

        shared_frame = ttk.Frame(drive_frame)
        shared_frame.pack(fill=tk.X, pady=5)
        ttk.Label(shared_frame, text="Include Shared Drives:").pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(shared_frame, variable=self.include_shared_drives_var).pack(side=tk.LEFT, padx=5)

    def _add_text_field(self, parent, label_text, variable):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        ttk.Label(frame, text=label_text).pack(side=tk.LEFT, padx=5)
        ttk.Entry(frame, textvariable=variable, width=70).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=1)

    def get_config(self):
        folder_ids = [item.strip() for item in self.folder_ids_var.get().split(",") if item.strip()]
        return {
            "google_drive": {
                "enabled": self.enabled_var.get(),
                "client_secret_file": self.client_secret_var.get().strip(),
                "token_file": self.token_file_var.get().strip(),
                "redirect_uri": self.redirect_uri_var.get().strip(),
                "folder_ids": folder_ids,
                "include_shared_drives": self.include_shared_drives_var.get(),
            }
        }
