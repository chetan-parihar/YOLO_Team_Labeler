import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import requests
import io
import json
import os

class NetworkClientApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Labeling Client - Enhanced")
        self.root.geometry("1200x800")
        
        # --- Network State ---
        self.server_url = ""
        self.user_name = ""
        self.current_image_name = None
        self.is_connected = False
        
        # History
        self.image_history = [] 
        self.history_index = -1 
        
        # --- App State ---
        self.labels = {} 
        self.current_label = "object"
        self.label_list = ["object"]
        self.label_colors = {"object": "#FF0000"}
        self.raw_image = None
        
        # Interaction State
        self.edit_mode = False
        self.drag_data = {
            "x": 0, "y": 0, 
            "item": None, 
            "mode": None,       # 'create', 'move', 'resize'
            "box_index": None,  # Index of box being edited
            "resize_handle": None # 'tl', 'tr', 'bl', 'br'
        }
        
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.selected_bbox_index = None

        self.auto_label_enabled = False

        self.setup_ui()
        self.setup_bindings()
        self.root.after(100, self.connect_dialog)

    def setup_ui(self):
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.center_pane = tk.Frame(self.main_container, bg="#2e2e2e")
        self.center_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sidebar = tk.Frame(self.main_container, width=250, bg="#f0f0f0", relief=tk.RAISED, borderwidth=1)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(self.center_pane, bg="#2e2e2e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.status_var = tk.StringVar(value="Not Connected")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Controls
        tk.Label(self.sidebar, text="Connection", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(pady=(10, 5))
        self.connect_btn = tk.Button(self.sidebar, text="Connect", command=self.connect_dialog, bg="lightblue")
        self.connect_btn.pack(pady=2, padx=10, fill=tk.X)
        self.user_lbl = tk.Label(self.sidebar, text="User: ???", bg="#f0f0f0", fg="gray")
        self.user_lbl.pack()

        tk.Frame(self.sidebar, height=2, bg="#cccccc").pack(fill=tk.X, pady=10)

        # Classes
        tk.Label(self.sidebar, text="Classes", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(pady=5)
        self.label_listbox = tk.Listbox(self.sidebar, selectmode=tk.SINGLE, height=8)
        self.label_listbox.pack(pady=5, padx=10, fill=tk.X)
        self.label_listbox.bind("<<ListboxSelect>>", self.on_label_select)
        
        btn_frame = tk.Frame(self.sidebar, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X, padx=10)
        tk.Button(btn_frame, text="+", width=3, command=self.add_label).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="-", width=3, command=self.delete_label).pack(side=tk.LEFT, padx=2)

        tk.Frame(self.sidebar, height=2, bg="#cccccc").pack(fill=tk.X, pady=10)
        self.model_btn = tk.Button(self.sidebar, text="Enable Server Auto-Label", command=self.toggle_model)
        self.model_btn.pack(pady=5, padx=10, fill=tk.X)

        tk.Frame(self.sidebar, height=2, bg="#cccccc").pack(fill=tk.X, pady=10)

        # Navigation
        nav_frame = tk.Frame(self.sidebar, bg="#f0f0f0")
        nav_frame.pack(padx=10, fill=tk.X, pady=5)
        
        self.prev_btn = tk.Button(nav_frame, text="< Prev (A)", command=self.go_back, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.next_btn = tk.Button(nav_frame, text="Next (D) >", command=self.submit_and_next, bg="#90ee90", state=tk.DISABLED)
        self.next_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=2)
        
        self.mode_label = tk.Label(self.sidebar, text="Hold Ctrl to Edit/Resize", fg="gray", bg="#f0f0f0")
        self.mode_label.pack(side=tk.BOTTOM, pady=10)

    def setup_bindings(self):
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Configure>", self.on_resize)
        
        self.root.bind("<Control_L>", self.enable_edit_mode)
        self.root.bind("<KeyRelease-Control_L>", self.disable_edit_mode)
        self.root.bind("<d>", lambda e: self.submit_and_next())
        self.root.bind("<a>", lambda e: self.go_back())
        self.root.bind("<w>", self.select_label_up)
        self.root.bind("<s>", self.select_label_down)

    # --- Network Logic ---
    def connect_dialog(self):
        ip = simpledialog.askstring("Connect", "Enter Server IP:", parent=self.root)
        if not ip: return
        user = simpledialog.askstring("Login", "Enter your Name:", parent=self.root)
        if not user: return
        self.server_url = f"http://{ip}:8000"
        self.user_name = user
        try:
            resp = requests.get(f"{self.server_url}/", timeout=3)
            if resp.status_code == 200:
                self.is_connected = True
                self.user_lbl.config(text=f"User: {self.user_name}", fg="green")
                self.status_var.set("Connected.")
                self.next_btn.config(state=tk.NORMAL)
                self.load_next_image()
            else: messagebox.showerror("Error", "Server error.")
        except: messagebox.showerror("Error", "Connection Failed.")

    def fetch_image_and_labels(self, endpoint, params={}):
        try:
            self.status_var.set("Fetching...")
            self.root.update_idletasks()
            resp = requests.get(f"{self.server_url}/{endpoint}", params=params)
            
            if resp.headers.get("content-type") == "application/json":
                data = resp.json()
                if data.get("status") == "done":
                    messagebox.showinfo("Done", "No more images!")
                    return False
                return False

            self.current_image_name = resp.headers.get("filename", "unknown.jpg")
            image_data = resp.content
            self.raw_image = Image.open(io.BytesIO(image_data))
            self.labels[self.current_image_name] = []
            
            lbl_resp = requests.get(f"{self.server_url}/get_current_labels", params={"image_name": self.current_image_name})
            if lbl_resp.status_code == 200:
                server_labels = lbl_resp.json().get("labels", [])
                for cls, box in server_labels:
                    if cls not in self.label_list:
                        self.label_list.append(cls)
                        self.label_colors[cls] = self.get_random_color()
                    self.labels[self.current_image_name].append((cls, box))
                self.update_label_listbox()
            
            self.display_image()
            if self.auto_label_enabled and not self.labels[self.current_image_name]:
                self.run_server_inference()
            self.status_var.set(f"Labeling: {self.current_image_name}")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Network Error: {e}")
            return False

    def load_next_image(self):
        if not self.is_connected: return
        success = self.fetch_image_and_labels("next_image", {"user_name": self.user_name})
        if success:
            if not self.image_history or self.image_history[-1] != self.current_image_name:
                self.image_history.append(self.current_image_name)
                self.history_index = len(self.image_history) - 1
            self.update_nav_buttons()

    def go_back(self, event=None):
        if not self.is_connected or self.history_index <= 0: return
        self.submit_labels_only() 
        self.history_index -= 1
        prev_img_name = self.image_history[self.history_index]
        success = self.fetch_image_and_labels("get_image_specific", {"filename": prev_img_name})
        if success: self.update_nav_buttons()

    def update_nav_buttons(self):
        self.prev_btn.config(state=tk.NORMAL if self.history_index > 0 else tk.DISABLED)

    def submit_labels_only(self):
        if not self.current_image_name: return
        current_data = self.labels.get(self.current_image_name, [])
        payload = {"image_name": self.current_image_name, "user_name": self.user_name, "labels": json.dumps(current_data)}
        try: requests.post(f"{self.server_url}/submit_label", data=payload)
        except: pass

    def submit_and_next(self, event=None):
        if not self.current_image_name: return
        self.submit_labels_only()
        if self.history_index < len(self.image_history) - 1:
            self.history_index += 1
            next_img_name = self.image_history[self.history_index]
            self.fetch_image_and_labels("get_image_specific", {"filename": next_img_name})
            self.update_nav_buttons()
        else:
            self.load_next_image()

    def run_server_inference(self):
        if not self.raw_image: return
        self.status_var.set("AI predicting...")
        self.root.update_idletasks()
        img_byte_arr = io.BytesIO()
        if self.raw_image.mode in ("RGBA", "P"): self.raw_image.convert("RGB").save(img_byte_arr, format='JPEG')
        else: self.raw_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        try:
            resp = requests.post(f"{self.server_url}/predict", files={'file': img_byte_arr})
            if resp.status_code == 200:
                predictions = resp.json()["predictions"]
                for label, bbox in predictions:
                    if label not in self.label_list:
                        self.label_list.append(label)
                        self.label_colors[label] = self.get_random_color()
                        self.update_label_listbox()
                    if not self.is_duplicate(bbox):
                        self.labels[self.current_image_name].append((label, bbox))
                self.redraw_labels()
                self.status_var.set(f"Labeling: {self.current_image_name} (AI Applied)")
        except Exception as e: print(e)

    # --- Canvas & Interaction Logic ---
    def display_image(self):
        if not self.raw_image: return
        self.canvas.delete("all")
        c_w, c_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        i_w, i_h = self.raw_image.size
        if c_w < 10: return
        ratio = min(c_w/i_w, c_h/i_h)
        new_w, new_h = int(i_w*ratio), int(i_h*ratio)
        self.scale_factor = ratio
        self.offset_x = (c_w-new_w)//2
        self.offset_y = (c_h-new_h)//2
        resized = self.raw_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)
        self.redraw_labels()

    def redraw_labels(self):
        self.canvas.delete("bbox")
        if not self.current_image_name: return
        for i, (label, bbox) in enumerate(self.labels.get(self.current_image_name, [])):
            x1, y1 = self.image_to_screen(bbox[0], bbox[1])
            x2, y2 = self.image_to_screen(bbox[2], bbox[3])
            color = self.label_colors.get(label, "red")
            
            # Highlight selected box
            width = 3 if (self.edit_mode and self.selected_bbox_index == i) else 2
            dash = (4,2) if (self.edit_mode and self.selected_bbox_index == i) else None
            
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, dash=dash, tags="bbox")
            self.canvas.create_text(x1, y1-12, text=label, fill=color, anchor=tk.SW, font=("Arial", 10, "bold"), tags="bbox")

    # --- Mouse Events ---
    def get_bbox_at(self, x, y):
        """Returns (index, handle_type) if x,y is near a box or its corners."""
        img_x, img_y = self.screen_to_image(x, y)
        boxes = self.labels.get(self.current_image_name, [])
        threshold = 10 / self.scale_factor # 10 pixel threshold
        
        # Search backwards (top boxes first)
        for i in range(len(boxes) - 1, -1, -1):
            _, bbox = boxes[i]
            x1, y1, x2, y2 = bbox
            
            # Check corners for resize
            if abs(img_x - x1) < threshold and abs(img_y - y1) < threshold: return i, 'tl'
            if abs(img_x - x2) < threshold and abs(img_y - y1) < threshold: return i, 'tr'
            if abs(img_x - x1) < threshold and abs(img_y - y2) < threshold: return i, 'bl'
            if abs(img_x - x2) < threshold and abs(img_y - y2) < threshold: return i, 'br'
            
            # Check inside for move
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                return i, 'move'
        return None, None

    def on_mouse_down(self, event):
        if not self.raw_image: return
        
        if self.edit_mode:
            # Check if we clicked an existing box to Move/Resize
            index, handle = self.get_bbox_at(event.x, event.y)
            
            if index is not None:
                self.selected_bbox_index = index
                self.drag_data["mode"] = 'resize' if handle in ['tl','tr','bl','br'] else 'move'
                self.drag_data["box_index"] = index
                self.drag_data["resize_handle"] = handle
                self.drag_data["x"] = event.x
                self.drag_data["y"] = event.y
                self.redraw_labels()
                return

        # Default: Create New Box
        self.drag_data["mode"] = 'create'
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        self.drag_data["item"] = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="yellow", width=2, dash=(2,2))

    def on_mouse_drag(self, event):
        if not self.raw_image: return
        
        mode = self.drag_data["mode"]
        
        if mode == 'create' and self.drag_data["item"]:
            self.canvas.coords(self.drag_data["item"], self.drag_data["x"], self.drag_data["y"], event.x, event.y)
            
        elif mode in ['move', 'resize']:
            idx = self.drag_data["box_index"]
            label, bbox = self.labels[self.current_image_name][idx]
            x1, y1, x2, y2 = bbox
            
            # Convert movement to image scale
            dx = (event.x - self.drag_data["x"]) / self.scale_factor
            dy = (event.y - self.drag_data["y"]) / self.scale_factor
            
            if mode == 'move':
                new_bbox = [x1+dx, y1+dy, x2+dx, y2+dy]
                self.labels[self.current_image_name][idx] = (label, new_bbox)
                self.drag_data["x"] = event.x
                self.drag_data["y"] = event.y
            
            elif mode == 'resize':
                handle = self.drag_data["resize_handle"]
                nx1, ny1, nx2, ny2 = x1, y1, x2, y2
                
                if 'l' in handle: nx1 += dx
                if 'r' in handle: nx2 += dx
                if 't' in handle: ny1 += dy
                if 'b' in handle: ny2 += dy
                
                # Normalize rect (handle negative size)
                self.labels[self.current_image_name][idx] = (label, [min(nx1,nx2), min(ny1,ny2), max(nx1,nx2), max(ny1,ny2)])
                self.drag_data["x"] = event.x
                self.drag_data["y"] = event.y
                
            self.redraw_labels()

    def on_mouse_up(self, event):
        mode = self.drag_data["mode"]
        
        if mode == 'create' and self.drag_data["item"]:
            self.canvas.delete(self.drag_data["item"])
            self.drag_data["item"] = None
            x1, y1 = self.screen_to_image(self.drag_data["x"], self.drag_data["y"])
            x2, y2 = self.screen_to_image(event.x, event.y)
            bbox = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]
            if abs(bbox[2]-bbox[0]) > 5 and abs(bbox[3]-bbox[1]) > 5:
                self.labels[self.current_image_name].append((self.current_label, bbox))
                self.selected_bbox_index = len(self.labels[self.current_image_name]) - 1
        
        self.drag_data["mode"] = None
        self.redraw_labels()

    def on_mouse_move(self, event):
        if not self.raw_image: return
        self.canvas.delete("crosshair")
        
        # Cursor feedback for Edit Mode
        if self.edit_mode:
            _, handle = self.get_bbox_at(event.x, event.y)
            if handle in ['tl', 'br']: self.canvas.config(cursor="tcross") # Diagonal
            elif handle in ['tr', 'bl']: self.canvas.config(cursor="tcross")
            elif handle == 'move': self.canvas.config(cursor="fleur") # Move icon
            else: self.canvas.config(cursor="arrow")
        else:
            self.canvas.config(cursor="cross") # Draw cursor

        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.create_line(0, event.y, w, event.y, fill="gray", dash=(4,4), tags="crosshair")
        self.canvas.create_line(event.x, 0, event.x, h, fill="gray", dash=(4,4), tags="crosshair")

    def on_right_click(self, event):
        if not self.raw_image: return
        img_x, img_y = self.screen_to_image(event.x, event.y)
        boxes = self.labels.get(self.current_image_name, [])
        for i, (label, bbox) in enumerate(boxes):
            if bbox[0] <= img_x <= bbox[2] and bbox[1] <= img_y <= bbox[3]:
                boxes.pop(i)
                self.selected_bbox_index = None
                self.redraw_labels()
                return

    # --- Utils ---
    def screen_to_image(self, sx, sy): return (sx - self.offset_x)/self.scale_factor, (sy - self.offset_y)/self.scale_factor
    def image_to_screen(self, ix, iy): return (ix * self.scale_factor)+self.offset_x, (iy * self.scale_factor)+self.offset_y
    def is_duplicate(self, new_box):
        for _, ex in self.labels.get(self.current_image_name, []):
             if abs(ex[0]-new_box[0]) < 2 and abs(ex[1]-new_box[1]) < 2: return True
        return False
    def on_resize(self, event): 
        if self.raw_image: self.display_image()
    def toggle_model(self):
        self.auto_label_enabled = not self.auto_label_enabled
        self.model_btn.config(bg="lightgreen" if self.auto_label_enabled else "#dddddd")
        if self.auto_label_enabled: self.run_server_inference()
    def add_label(self):
        name = simpledialog.askstring("Class", "New Class Name:")
        if name and name not in self.label_list:
            self.label_list.append(name)
            self.label_colors[name] = self.get_random_color()
            self.update_label_listbox()
    def delete_label(self):
        sel = self.label_listbox.curselection()
        if sel: self.label_list.remove(self.label_listbox.get(sel)); self.update_label_listbox()
    def update_label_listbox(self):
        self.label_listbox.delete(0, tk.END)
        for l in self.label_list:
            self.label_listbox.insert(tk.END, l)
            self.label_listbox.itemconfig(tk.END, {'bg': self.label_colors.get(l, "white")})
        if self.label_list: self.label_listbox.select_set(0); self.current_label = self.label_list[0]
    def on_label_select(self, event):
        sel = self.label_listbox.curselection()
        if sel: self.current_label = self.label_listbox.get(sel)
    def select_label_up(self, e):
        sel = self.label_listbox.curselection()
        if sel and sel[0]>0: self.label_listbox.selection_clear(0, tk.END); self.label_listbox.selection_set(sel[0]-1); self.current_label = self.label_listbox.get(sel[0]-1)
    def select_label_down(self, e):
        sel = self.label_listbox.curselection()
        if sel and sel[0]<self.label_listbox.size()-1: self.label_listbox.selection_clear(0, tk.END); self.label_listbox.selection_set(sel[0]+1); self.current_label = self.label_listbox.get(sel[0]+1)
    def enable_edit_mode(self, e): 
        self.edit_mode = True
        self.mode_label.config(text="Mode: EDIT (Drag box to move, corners to resize)", fg="red")
        self.redraw_labels()
    def disable_edit_mode(self, e): 
        self.edit_mode = False
        self.mode_label.config(text="Hold Ctrl to Edit/Resize", fg="gray")
        self.selected_bbox_index = None
        self.redraw_labels()
    def get_random_color(self): import random; return "#{:06x}".format(random.randint(0, 0xFFFFFF))

if __name__ == "__main__":
    root = tk.Tk()
    app = NetworkClientApp(root)
    root.mainloop()