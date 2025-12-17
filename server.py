import os
import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO
import uvicorn
from PIL import Image
import io
import shutil
import yaml
from datetime import datetime

# --- 1. SERVER STATE & API ---

class ServerState:
    def __init__(self):
        self.image_folder = ""
        self.label_folder = ""
        self.model_path = "yolov8n.pt"
        self.model = None
        self.conf_threshold = 0.25  # Default Confidence
        self.in_progress = {} # {user: image_name}
        self.app = FastAPI()
        self.log_callback = None 

    def log(self, message):
        print(message)
        if self.log_callback:
            self.log_callback(message)

    def load_model(self, path):
        try:
            self.log(f"Loading model: {path}...")
            # Load new model
            new_model = YOLO(path)
            # Update state
            self.model = new_model
            self.model_path = path
            self.log(f"SUCCESS: Switched to {os.path.basename(path)}")
            return True
        except Exception as e:
            self.log(f"Error loading model: {e}")
            return False

server_state = ServerState()

# --- FASTAPI ENDPOINTS ---

@server_state.app.get("/")
def health_check():
    return {"status": "online", "model": os.path.basename(server_state.model_path)}

@server_state.app.get("/next_image")
def next_image(user_name: str):
    if not server_state.image_folder:
        return {"status": "error", "message": "Server not configured"}

    completed = set()
    if os.path.exists(server_state.label_folder):
        for f in os.listdir(server_state.label_folder):
            if f.endswith(".txt"):
                completed.add(os.path.splitext(f)[0])

    all_images = [f for f in os.listdir(server_state.image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    assigned = set(server_state.in_progress.values())
    selected = None

    if user_name in server_state.in_progress:
        base = os.path.splitext(server_state.in_progress[user_name])[0]
        if base not in completed:
            selected = server_state.in_progress[user_name]

    if not selected:
        for img in all_images:
            base = os.path.splitext(img)[0]
            if base not in completed and img not in assigned:
                selected = img
                break
    
    if not selected:
        return {"status": "done"}

    server_state.in_progress[user_name] = selected
    server_state.log(f"Assigning {selected} to {user_name}")
    return FileResponse(os.path.join(server_state.image_folder, selected), headers={"filename": selected})

@server_state.app.get("/get_image_specific")
def get_image_specific(filename: str):
    file_path = os.path.join(server_state.image_folder, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, headers={"filename": filename})
    return {"status": "error", "message": "File not found"}

@server_state.app.get("/get_current_labels")
def get_current_labels(image_name: str):
    txt_name = os.path.splitext(image_name)[0] + ".txt"
    txt_path = os.path.join(server_state.label_folder, txt_name)
    labels = []
    if os.path.exists(txt_path):
        try:
            img_path = os.path.join(server_state.image_folder, image_name)
            with Image.open(img_path) as img:
                w, h = img.size
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_name = parts[0]
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = (cx - bw/2) * w
                        y1 = (cy - bh/2) * h
                        x2 = (cx + bw/2) * w
                        y2 = (cy + bh/2) * h
                        labels.append((cls_name, [x1, y1, x2, y2]))
        except Exception as e:
            print(f"Error reading labels: {e}")
    return {"labels": labels}

@server_state.app.post("/submit_label")
async def submit_label(image_name: str = Form(...), user_name: str = Form(...), labels: str = Form(...)):
    try:
        data = json.loads(labels)
        txt_path = os.path.join(server_state.label_folder, os.path.splitext(image_name)[0] + ".txt")
        img_path = os.path.join(server_state.image_folder, image_name)

        if not os.path.exists(img_path):
             return {"status": "error", "message": "Image source not found"}

        with Image.open(img_path) as img:
            w, h = img.size

        with open(txt_path, "w") as f:
            for label, bbox in data:
                x1, y1, x2, y2 = bbox
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{label} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        if user_name in server_state.in_progress and server_state.in_progress[user_name] == image_name:
            del server_state.in_progress[user_name]
            
        server_state.log(f"Saved labels for {image_name} by {user_name}")
        return {"status": "success"}
    except Exception as e:
        server_state.log(f"Save error: {e}")
        raise HTTPException(500, str(e))

@server_state.app.post("/predict")
async def predict(file: UploadFile):
    if not server_state.model:
        return {"error": "No model loaded"}
    
    img_data = await file.read()
    img = Image.open(io.BytesIO(img_data))
    
    # Use CURRENT model AND Confidence
    conf = float(server_state.conf_threshold)
    results = server_state.model(img, conf=conf)
    
    preds = []
    for r in results:
        for box in r.boxes:
            preds.append((server_state.model.names[int(box.cls[0])], box.xyxy[0].tolist()))
    return {"predictions": preds}

# --- 2. GUI IMPLEMENTATION ---

class ServerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Server Control Panel")
        self.root.geometry("950x750")
        
        server_state.log_callback = self.append_log

        # --- Top Section: Config ---
        config_frame = tk.LabelFrame(root, text="Server Configuration", padx=10, pady=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)

        # 1. Image Folder
        tk.Label(config_frame, text="Images Folder:").grid(row=0, column=0, sticky="w")
        self.folder_var = tk.StringVar()
        tk.Entry(config_frame, textvariable=self.folder_var, width=40).grid(row=0, column=1, padx=5)
        tk.Button(config_frame, text="Browse", command=self.select_folder).grid(row=0, column=2)

        # 2. Model Selection
        tk.Label(config_frame, text="Model Path:").grid(row=1, column=0, sticky="w")
        self.model_var = tk.StringVar(value="yolov8n.pt")
        self.model_entry = tk.Entry(config_frame, textvariable=self.model_var, width=40)
        self.model_entry.grid(row=1, column=1, padx=5)
        
        self.select_model_btn = tk.Button(config_frame, text="Select Model", command=self.browse_model_file)
        self.select_model_btn.grid(row=1, column=2, padx=2)
        
        self.switch_btn = tk.Button(config_frame, text="Switch Model", bg="lightblue", command=self.switch_model)
        self.switch_btn.grid(row=1, column=3, padx=2)

        # 3. Confidence Slider
        tk.Label(config_frame, text="AI Confidence:").grid(row=2, column=0, sticky="w")
        self.conf_scale = tk.Scale(config_frame, from_=0.05, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=300, command=self.update_conf)
        self.conf_scale.set(0.25)
        self.conf_scale.grid(row=2, column=1, columnspan=2, sticky="w", pady=5)

        # 4. Start Server (Spans 3 rows now)
        self.start_btn = tk.Button(config_frame, text="START SERVER", bg="lightgreen", font=("Arial", 10, "bold"), command=self.start_server_thread)
        self.start_btn.grid(row=0, column=4, rowspan=3, padx=10, sticky="nsew")

        # --- Tabs ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab 1: Logs
        self.log_tab = tk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text="Server Logs")
        self.log_text = scrolledtext.ScrolledText(self.log_tab, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Tab 2: Export
        self.export_tab = tk.Frame(self.notebook)
        self.notebook.add(self.export_tab, text="Export Data")
        self.setup_export_tab()

        # Tab 3: Training
        self.train_tab = tk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text="Training")
        self.setup_training_tab()

    def append_log(self, msg):
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see(tk.END)

    def select_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.folder_var.set(path)
            server_state.image_folder = path
            server_state.label_folder = os.path.join(path, "labels_collected")
            os.makedirs(server_state.label_folder, exist_ok=True)
            self.append_log(f"Root: {path}")

    def update_conf(self, val):
        server_state.conf_threshold = float(val)

    # --- Model Management ---

    def browse_model_file(self):
        filename = filedialog.askopenfilename(filetypes=[("YOLO Models", "*.pt")])
        if filename:
            self.model_var.set(filename)

    def switch_model(self):
        path = self.model_var.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "File not found or invalid path.")
            return
        
        success = server_state.load_model(path)
        if success:
            messagebox.showinfo("Success", f"Server is now using:\n{os.path.basename(path)}")
        else:
            messagebox.showerror("Error", "Failed to load model. Check logs.")

    def start_server_thread(self):
        if not server_state.image_folder:
            messagebox.showerror("Error", "Select Image Folder first.")
            return
        
        initial_model = self.model_var.get() or "yolov8n.pt"
        if os.path.exists(initial_model):
            server_state.load_model(initial_model)
        else:
            self.append_log("Warning: Initial model file not found. Server started without model.")

        self.start_btn.config(state=tk.DISABLED, text="RUNNING...")
        threading.Thread(target=lambda: uvicorn.run(server_state.app, host="0.0.0.0", port=8000, log_level="error"), daemon=True).start()

    # --- Export ---
    def setup_export_tab(self):
        f = tk.Frame(self.export_tab, padx=20, pady=20)
        f.pack(fill=tk.BOTH)
        tk.Label(f, text="Export Labeled Dataset", font=("Arial", 14, "bold")).pack(pady=10)
        tk.Button(f, text="Export Now", bg="lightblue", font=("Arial", 12), command=self.export_data).pack(pady=20)

    def export_data(self):
        if not server_state.image_folder: return
        export_dir = filedialog.askdirectory(title="Select Folder to Export To")
        if not export_dir: return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = os.path.join(export_dir, f"dataset_export_{ts}")
        try:
            self.create_dataset_structure(target_path)
            messagebox.showinfo("Success", f"Exported to:\n{target_path}")
            self.append_log(f"Exported dataset to {target_path}")
        except Exception as e:
            messagebox.showerror("Failed", str(e))

    def create_dataset_structure(self, target_path):
        os.makedirs(os.path.join(target_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(target_path, "labels"), exist_ok=True)
        
        classes = set()
        file_pairs = [] 

        if not os.path.exists(server_state.label_folder):
             return "", 0

        for txt_file in os.listdir(server_state.label_folder):
            if not txt_file.endswith(".txt"): continue
            txt_full = os.path.join(server_state.label_folder, txt_file)
            base_name = os.path.splitext(txt_file)[0]
            
            img_full = None
            for ext in [".jpg", ".png", ".jpeg"]:
                potential = os.path.join(server_state.image_folder, base_name + ext)
                if os.path.exists(potential):
                    img_full = potential
                    break
            
            if img_full:
                file_pairs.append((img_full, txt_full))
                with open(txt_full, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if parts: classes.add(parts[0])

        class_list = sorted(list(classes))
        
        count = 0
        for img_p, txt_p in file_pairs:
            fname = os.path.basename(img_p)
            tname = os.path.basename(txt_p)
            shutil.copy(img_p, os.path.join(target_path, "images", fname))
            with open(txt_p, 'r') as source_t:
                with open(os.path.join(target_path, "labels", tname), 'w') as dest_t:
                    for line in source_t:
                        parts = line.split()
                        if parts[0] in class_list:
                            idx = class_list.index(parts[0])
                            dest_t.write(f"{idx} " + " ".join(parts[1:]) + "\n")
            count += 1
        
        yaml_content = {'path': os.path.abspath(target_path), 'train': 'images', 'val': 'images', 'nc': len(class_list), 'names': class_list}
        yaml_path = os.path.join(target_path, "data.yaml")
        with open(yaml_path, 'w') as yf:
            yaml.dump(yaml_content, yf)
            
        return yaml_path, count

    # --- Training ---
    def setup_training_tab(self):
        f = tk.Frame(self.train_tab, padx=20, pady=20)
        f.pack(fill=tk.BOTH)
        tk.Label(f, text="Train New Model on Server", font=("Arial", 14, "bold")).pack(pady=10)
        
        grid = tk.Frame(f)
        grid.pack(pady=10)
        tk.Label(grid, text="Epochs:").grid(row=0, column=0); 
        self.epochs_ent = tk.Entry(grid, width=5); self.epochs_ent.insert(0,"10"); self.epochs_ent.grid(row=0,column=1)
        tk.Label(grid, text="Batch:").grid(row=0, column=2); 
        self.batch_ent = tk.Entry(grid, width=5); self.batch_ent.insert(0,"16"); self.batch_ent.grid(row=0,column=3)

        self.train_btn = tk.Button(f, text="Start Training", bg="orange", font=("Arial", 12), command=self.start_training_process)
        self.train_btn.pack(pady=20)
        self.train_status = tk.Label(f, text="Ready", fg="gray")
        self.train_status.pack()

    def start_training_process(self):
        if not server_state.image_folder:
            messagebox.showerror("Error", "No image folder selected.")
            return

        try:
            epochs = int(self.epochs_ent.get())
            batch = int(self.batch_ent.get())
        except ValueError:
            messagebox.showerror("Error", "Epochs and Batch must be numbers.")
            return

        self.train_btn.config(state=tk.DISABLED, text="Training in Progress...")
        self.train_status.config(text="Preparing Dataset...", fg="blue")
        
        threading.Thread(target=self.run_training_logic, args=(epochs, batch)).start()

    def run_training_logic(self, epochs, batch):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            train_dir = os.path.join(server_state.image_folder, f"server_train_{ts}")
            
            self.append_log(f"Preparing data in: {train_dir}")
            yaml_path, count = self.create_dataset_structure(train_dir)
            
            if count == 0:
                self.append_log("No labeled data found.")
                self.root.after(0, lambda: self.reset_train_ui("No Data"))
                return

            self.append_log(f"Starting training ({epochs} epochs)...")
            
            base_model = server_state.model_path if server_state.model_path else "yolov8n.pt"
            self.append_log(f"Base model: {os.path.basename(base_model)}")
            
            train_model = YOLO(base_model)
            results = train_model.train(data=yaml_path, epochs=epochs, batch=batch, imgsz=640)
            
            new_model_path = ""
            if hasattr(results, 'save_dir'):
                new_model_path = os.path.join(results.save_dir, "weights", "best.pt")
            
            self.append_log(f"TRAINING COMPLETE.")
            self.append_log(f"New Model: {new_model_path}")

            self.root.after(0, lambda: self.training_finished_ui(new_model_path))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.append_log(f"Training Error: {e}")
            self.root.after(0, lambda: self.reset_train_ui("Error"))

    def training_finished_ui(self, new_model_path):
        self.reset_train_ui("Training Complete")
        
        if new_model_path and os.path.exists(new_model_path):
            ans = messagebox.askyesno("Training Complete", 
                f"Training finished successfully!\n\nNew model saved at:\n{new_model_path}\n\nDo you want to switch to this model now?")
            
            if ans:
                self.model_var.set(new_model_path)
                self.switch_model()

    def reset_train_ui(self, status_text):
        self.train_btn.config(state=tk.NORMAL, text="Start Training")
        self.train_status.config(text=status_text, fg="black")

if __name__ == "__main__":
    root = tk.Tk()
    app = ServerGUI(root)
    root.mainloop()