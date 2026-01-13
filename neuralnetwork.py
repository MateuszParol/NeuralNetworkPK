import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Neural Network Configuration
INPUT_SIZE = 784
HIDDEN_SIZE = 128
OUTPUT_SIZE = 36
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class NeuralNetwork:
    def __init__(self):
        # Xavier initialization for sigmoid
        self.w1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * np.sqrt(1.0 / INPUT_SIZE)
        self.b1 = np.zeros((1, HIDDEN_SIZE))
        self.w2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * np.sqrt(1.0 / HIDDEN_SIZE)
        self.b2 = np.zeros((1, OUTPUT_SIZE))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def train(self, x, y, lr=0.1):
        output = self.forward(x)

        # Output layer gradient
        d2 = output - y
        dw2 = np.dot(self.a1.T, d2)
        db2 = np.sum(d2, axis=0, keepdims=True)

        # Hidden layer gradient
        d1 = np.dot(d2, self.w2.T) * self.sigmoid_derivative(self.a1)
        dw1 = np.dot(x.T, d1)
        db1 = np.sum(d1, axis=0, keepdims=True)

        # Update weights
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1

        # Clip to prevent explosion
        self.w1 = np.clip(self.w1, -10, 10)
        self.w2 = np.clip(self.w2, -10, 10)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network - Character Recognition")
        self.root.geometry("1200x700")
        self.root.configure(bg="#F5F5F5")

        self.nn = NeuralNetwork()
        self.canvas_data = np.zeros((28, 28))

        self.setup_ui()

    def setup_ui(self):
        # Main container
        main = tk.Frame(self.root, bg="#F5F5F5")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left panel - Drawing
        left = tk.Frame(main, bg="white", relief="flat", bd=0)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        tk.Label(left, text="Draw", font=("Segoe UI", 16, "bold"),
                bg="white", fg="#333").pack(pady=15)

        self.canvas = tk.Canvas(left, width=280, height=280, bg="white",
                               highlightthickness=1, highlightbackground="#DDD")
        self.canvas.pack(padx=20, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.recognize())

        btn_clear = tk.Button(left, text="Clear", command=self.clear,
                             bg="#2196F3", fg="white", font=("Segoe UI", 11),
                             relief="flat", cursor="hand2", padx=30, pady=10)
        btn_clear.pack(pady=10)

        # Center panel - Visualization
        center = tk.Frame(main, bg="white", relief="flat")
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        tk.Label(center, text="Network Visualization", font=("Segoe UI", 16, "bold"),
                bg="white", fg="#333").pack(pady=15)

        self.vis_canvas = tk.Canvas(center, bg="white", highlightthickness=1,
                                    highlightbackground="#DDD")
        self.vis_canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Right panel - Results & Training
        right = tk.Frame(main, bg="white", relief="flat", width=280)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        right.pack_propagate(False)

        tk.Label(right, text="Result", font=("Segoe UI", 16, "bold"),
                bg="white", fg="#333").pack(pady=15)

        self.result_label = tk.Label(right, text="?", font=("Segoe UI", 72, "bold"),
                                     bg="white", fg="#2196F3")
        self.result_label.pack(pady=20)

        self.confidence_label = tk.Label(right, text="Confidence: 0%",
                                        font=("Segoe UI", 12), bg="white", fg="#666")
        self.confidence_label.pack()

        tk.Frame(right, height=2, bg="#EEE").pack(fill=tk.X, pady=30, padx=20)

        tk.Label(right, text="Training", font=("Segoe UI", 14, "bold"),
                bg="white", fg="#333").pack(pady=10)

        self.train_btn = tk.Button(right, text="Load & Train", command=self.start_training,
                                   bg="#4CAF50", fg="white", font=("Segoe UI", 11),
                                   relief="flat", cursor="hand2", padx=20, pady=10)
        self.train_btn.pack(pady=10)

        self.progress = ttk.Progressbar(right, length=240, mode="determinate")
        self.progress.pack(pady=10)

        self.status_label = tk.Label(right, text="Ready", font=("Segoe UI", 9),
                                     bg="white", fg="#666", wraplength=240)
        self.status_label.pack(pady=10)

        # Initialize visualization
        self.root.after(100, self.init_visualization)

    def init_visualization(self):
        w = self.vis_canvas.winfo_width()
        h = self.vis_canvas.winfo_height()

        # Input layer (20 nodes)
        self.input_nodes = [(30, h * (i+1) / 21) for i in range(20)]
        for x, y in self.input_nodes:
            self.vis_canvas.create_oval(x-3, y-3, x+3, y+3, fill="#DDD", outline="")

        # Hidden layer (32 nodes in grid)
        self.hidden_nodes = []
        for i in range(32):
            row, col = i % 8, i // 8
            x = w/2 - 60 + col * 30
            y = 40 + row * (h-80) / 8
            self.hidden_nodes.append((x, y))
            self.vis_canvas.create_oval(x-4, y-4, x+4, y+4, fill="#DDD",
                                       outline="", tags=f"h{i}")

        # Output layer (36 nodes)
        self.output_nodes = []
        for i in range(OUTPUT_SIZE):
            x = w - 60
            y = h * (i+1) / 37
            self.output_nodes.append((x, y))
            self.vis_canvas.create_oval(x-4, y-4, x+4, y+4, fill="#DDD",
                                       outline="", tags=f"o{i}")
            self.vis_canvas.create_text(x+20, y, text=CHARS[i], fill="#999",
                                       font=("Segoe UI", 9), tags=f"t{i}")

    def update_visualization(self):
        self.vis_canvas.delete("line")

        # Get activations
        hidden = self.nn.a1[0]
        output = self.nn.a2[0]

        # Show top 20 active hidden nodes
        top_hidden = np.argsort(hidden)[-20:]
        winner = np.argmax(output)

        for i in top_hidden:
            if i >= len(self.hidden_nodes):
                continue

            act = hidden[i]
            intensity = int(act * 200) + 55
            color = f"#{intensity:02x}{intensity//2:02x}{255:02x}"
            self.vis_canvas.itemconfig(f"h{i}", fill=color)

            # Draw connection to winner
            if i < HIDDEN_SIZE and winner < OUTPUT_SIZE:
                weight = self.nn.w2[i][winner]
                width = abs(weight) * 2
                if width > 0.3:
                    line_color = "#4CAF50" if weight > 0 else "#F44336"
                    x1, y1 = self.hidden_nodes[i]
                    x2, y2 = self.output_nodes[winner]
                    self.vis_canvas.create_line(x1, y1, x2, y2, fill=line_color,
                                               width=width, tags="line")

        # Highlight winner
        for i in range(OUTPUT_SIZE):
            if i == winner:
                self.vis_canvas.itemconfig(f"o{i}", fill="#2196F3")
                self.vis_canvas.itemconfig(f"t{i}", fill="#2196F3",
                                          font=("Segoe UI", 10, "bold"))
            else:
                self.vis_canvas.itemconfig(f"o{i}", fill="#DDD")
                self.vis_canvas.itemconfig(f"t{i}", fill="#999",
                                          font=("Segoe UI", 9))

    def draw(self, event):
        x, y = event.x, event.y
        r = 10
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="")

        # Update canvas data
        px, py = int(x * 28 / 280), int(y * 28 / 280)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = px + dx, py + dy
                if 0 <= nx < 28 and 0 <= ny < 28:
                    self.canvas_data[ny, nx] = min(1.0, self.canvas_data[ny, nx] + 0.5)

    def clear(self):
        self.canvas.delete("all")
        self.canvas_data = np.zeros((28, 28))
        self.result_label.config(text="?")
        self.confidence_label.config(text="Confidence: 0%")

    def recognize(self):
        x = self.canvas_data.flatten().reshape(1, -1)
        output = self.nn.forward(x)

        pred = np.argmax(output)
        conf = output[0][pred] * 100

        self.result_label.config(text=CHARS[pred])
        self.confidence_label.config(text=f"Confidence: {conf:.1f}%")

        self.update_visualization()

    def start_training(self):
        if not HAS_PANDAS:
            messagebox.showerror("Error", "pandas is required for training.\nInstall: pip install pandas")
            return

        file = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not file:
            return

        thread = threading.Thread(target=self.train_network, args=(file,))
        thread.daemon = True
        thread.start()

    def train_network(self, file):
        try:
            self.status_label.config(text="Loading data...")
            self.train_btn.config(state="disabled")

            df = pd.read_csv(file, header=None)
            labels = df.iloc[:, 0].values
            pixels = df.iloc[:, 1:].values

            # Filter to 0-35
            mask = labels < OUTPUT_SIZE
            labels = labels[mask]
            pixels = pixels[mask]

            # Shuffle
            idx = np.random.permutation(len(labels))
            labels = labels[idx]
            pixels = pixels[idx]

            # Normalize
            pixels = pixels / 255.0

            # Use subset
            n_samples = min(15000, len(labels))
            labels = labels[:n_samples]
            pixels = pixels[:n_samples]

            self.progress["maximum"] = n_samples

            # Train
            for i in range(n_samples):
                x = pixels[i].reshape(1, -1)
                y = np.zeros((1, OUTPUT_SIZE))
                y[0][labels[i]] = 1

                self.nn.train(x, y, lr=0.05)

                if i % 100 == 0:
                    self.progress["value"] = i
                    self.status_label.config(text=f"Training: {i}/{n_samples}")
                    self.root.update()

            self.progress["value"] = n_samples
            self.status_label.config(text=f"Training complete! ({n_samples} samples)")
            self.train_btn.config(state="normal")

        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            self.train_btn.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
