import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import threading
import time
import pickle
import os

# Sprawdzamy czy mamy pandas
try:
    import pandas as pd
    PANDAS_DOSTEPNY = True
except ImportError:
    PANDAS_DOSTEPNY = False
    print("BRAK PANDAS! Zainstaluj: pip install pandas")

# --- KONFIGURACJA ---
THEME = {
    "bg": "#0a0e27",
    "panel": "#151932",
    "panel_light": "#1e2447",
    "accent": "#00d9ff",
    "accent2": "#7b2ff7",
    "text": "#e8f4f8",
    "synapse_pos": "#00ff88",
    "synapse_neg": "#ff2e63",
    "neuron_active": "#00d9ff",
    "neuron_inactive": "#2a2e4a",
    "button_bg": "#7b2ff7",
    "button_hover": "#9d4efa"
}

ROZMIAR = 28
WEJSCIA = ROZMIAR * ROZMIAR
UKRYTE1 = 256  # Zwiększona warstwa ukryta
UKRYTE2 = 128  # Druga warstwa ukryta dla lepszej dokładności
WYJSCIA = 36
ZNAKI = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

class SiecNeuronowa:
    """Zaawansowana sieć neuronowa z 3 warstwami ukrytymi dla wysokiej dokładności"""
    def __init__(self):
        np.random.seed(42)
        # Inicjalizacja He dla ReLU
        self.w1 = np.random.randn(WEJSCIA, UKRYTE1) * np.sqrt(2/WEJSCIA)
        self.w2 = np.random.randn(UKRYTE1, UKRYTE2) * np.sqrt(2/UKRYTE1)
        self.w3 = np.random.randn(UKRYTE2, WYJSCIA) * np.sqrt(2/UKRYTE2)

        self.b1 = np.zeros((1, UKRYTE1))
        self.b2 = np.zeros((1, UKRYTE2))
        self.b3 = np.zeros((1, WYJSCIA))

        # Bufory aktywacji dla wizualizacji
        self.ukryta1 = np.zeros((1, UKRYTE1))
        self.ukryta2 = np.zeros((1, UKRYTE2))
        self.wyjscie = np.zeros((1, WYJSCIA))

        # Animacja - zapisujemy historię aktywacji
        self.historia_ukryta1 = []
        self.historia_ukryta2 = []
        self.historia_wyjscie = []

    def relu(self, x):
        """ReLU activation dla lepszej wydajności"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Pochodna ReLU"""
        return (x > 0).astype(float)

    def sigmoid(self, x):
        """Sigmoid dla warstwy wyjściowej"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        """Pochodna sigmoid"""
        return x * (1 - x)

    def softmax(self, x):
        """Softmax dla lepszych prawdopodobieństw"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x, record_history=False):
        """Forward pass z opcjonalnym zapisem historii dla animacji"""
        # Warstwa 1
        z1 = np.dot(x, self.w1) + self.b1
        self.ukryta1 = self.relu(z1)

        # Warstwa 2
        z2 = np.dot(self.ukryta1, self.w2) + self.b2
        self.ukryta2 = self.relu(z2)

        # Warstwa wyjściowa
        z3 = np.dot(self.ukryta2, self.w3) + self.b3
        self.wyjscie = self.softmax(z3)

        # Zapisz historię dla animacji
        if record_history:
            self.historia_ukryta1.append(self.ukryta1.copy())
            self.historia_ukryta2.append(self.ukryta2.copy())
            self.historia_wyjscie.append(self.wyjscie.copy())

            # Ogranicz historię do ostatnich 10 klatek
            if len(self.historia_ukryta1) > 10:
                self.historia_ukryta1.pop(0)
                self.historia_ukryta2.pop(0)
                self.historia_wyjscie.pop(0)

        return self.wyjscie

    def train(self, x, y, lr=0.01, l2_lambda=0.0001):
        """Trening z L2 regularizacją i dropout simulation"""
        # Forward pass
        wyjscie = self.forward(x)
        batch_size = x.shape[0]

        # Backpropagation
        # Warstwa wyjściowa
        delta3 = wyjscie - y
        dw3 = np.dot(self.ukryta2.T, delta3) / batch_size
        db3 = np.sum(delta3, axis=0, keepdims=True) / batch_size

        # Warstwa 2
        delta2 = np.dot(delta3, self.w3.T) * self.relu_derivative(self.ukryta2)
        dw2 = np.dot(self.ukryta1.T, delta2) / batch_size
        db2 = np.sum(delta2, axis=0, keepdims=True) / batch_size

        # Warstwa 1
        delta1 = np.dot(delta2, self.w2.T) * self.relu_derivative(self.ukryta1)
        dw1 = np.dot(x.T, delta1) / batch_size
        db1 = np.sum(delta1, axis=0, keepdims=True) / batch_size

        # Aktualizacja wag z L2 regularizacją
        self.w3 -= lr * (dw3 + l2_lambda * self.w3)
        self.b3 -= lr * db3
        self.w2 -= lr * (dw2 + l2_lambda * self.w2)
        self.b2 -= lr * db2
        self.w1 -= lr * (dw1 + l2_lambda * self.w1)
        self.b1 -= lr * db1

        # Gradient clipping dla stabilności
        self.w1 = np.clip(self.w1, -10, 10)
        self.w2 = np.clip(self.w2, -10, 10)
        self.w3 = np.clip(self.w3, -10, 10)

        return np.mean((y - wyjscie) ** 2)

    def save_model(self, filepath):
        """Zapisz wagi modelu"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'w1': self.w1, 'w2': self.w2, 'w3': self.w3,
                'b1': self.b1, 'b2': self.b2, 'b3': self.b3
            }, f)

    def load_model(self, filepath):
        """Wczytaj wagi modelu"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.w1 = data['w1']
                self.w2 = data['w2']
                self.w3 = data['w3']
                self.b1 = data['b1']
                self.b2 = data['b2']
                self.b3 = data['b3']
            return True
        return False


class AnimowanaWizualizacja:
    """Klasa obsługująca zaawansowane animacje sieci neuronowej"""
    def __init__(self, canvas, siec):
        self.canvas = canvas
        self.siec = siec
        self.animacja_aktywna = False
        self.klatka_animacji = 0
        self.particles = []  # Cząsteczki pokazujące przepływ danych

    def dodaj_particle(self, x1, y1, x2, y2, kolor):
        """Dodaj animowaną cząsteczkę pokazującą przepływ sygnału"""
        self.particles.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'postep': 0.0, 'kolor': kolor, 'zycie': 30
        })

    def aktualizuj_particles(self):
        """Aktualizuj i rysuj cząsteczki"""
        nowe_particles = []
        for p in self.particles:
            p['postep'] += 0.1
            p['zycie'] -= 1

            if p['postep'] <= 1.0 and p['zycie'] > 0:
                x = p['x1'] + (p['x2'] - p['x1']) * p['postep']
                y = p['y1'] + (p['y2'] - p['y1']) * p['postep']

                # Rysuj świecącą cząsteczkę
                rozmiar = 4 * (1 - p['postep'])
                alpha = int(255 * (1 - p['postep']))
                self.canvas.create_oval(
                    x-rozmiar, y-rozmiar, x+rozmiar, y+rozmiar,
                    fill=p['kolor'], outline="", tags="particle"
                )
                nowe_particles.append(p)

        self.particles = nowe_particles


class AplikacjaNeural:
    """Główna aplikacja z zaawansowaną wizualizacją i animacjami"""
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Neural Network OCR v9.0 - Advanced Visualization")
        self.root.geometry("1600x900")
        self.root.configure(bg=THEME["bg"])

        self.siec = SiecNeuronowa()
        self.plansza = np.zeros((ROZMIAR, ROZMIAR))
        self.rysowanie_aktywne = False
        self.animacja_aktywna = False

        # Próba wczytania zapisanego modelu
        if self.siec.load_model("model_ocr.pkl"):
            print("✓ Wczytano zapisany model!")

        self.buduj_interfejs()
        self.root.after(100, self.inicjuj_wizualizacje)
        self.root.after(500, self.petla_animacji)

    def buduj_interfejs(self):
        """Buduje nowoczesny interfejs z gradientami i animacjami"""
        # Header
        header = tk.Frame(self.root, bg=THEME["panel"], height=80)
        header.pack(fill=tk.X, padx=0, pady=0)
        header.pack_propagate(False)

        tk.Label(header, text="🧠 NEURAL NETWORK OCR",
                fg=THEME["accent"], bg=THEME["panel"],
                font=("Segoe UI", 24, "bold")).pack(side=tk.LEFT, padx=30, pady=20)

        tk.Label(header, text="Deep Learning | Character Recognition | Real-time Visualization",
                fg=THEME["text"], bg=THEME["panel"],
                font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=10)

        # Przyciski zapisu/wczytania
        btn_frame = tk.Frame(header, bg=THEME["panel"])
        btn_frame.pack(side=tk.RIGHT, padx=30)

        self.create_button(btn_frame, "💾 SAVE", self.zapisz_model, width=12).pack(side=tk.LEFT, padx=5)
        self.create_button(btn_frame, "📂 LOAD", self.wczytaj_model, width=12).pack(side=tk.LEFT, padx=5)

        # Main container
        main = tk.Frame(self.root, bg=THEME["bg"])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # LEWY PANEL - RYSOWANIE
        p1 = self.create_panel(main, "🎨 DRAWING CANVAS", width=380)
        p1.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Canvas do rysowania
        canvas_container = tk.Frame(p1, bg=THEME["bg"], highlightthickness=2,
                                    highlightbackground=THEME["accent"])
        canvas_container.pack(pady=20, padx=20)

        self.canvas = tk.Canvas(canvas_container, width=320, height=320,
                               bg="#000000", highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.rysuj)
        self.canvas.bind("<Button-1>", self.rozpocznij_rysowanie)
        self.canvas.bind("<ButtonRelease-1>", self.koniec_rysowania)

        # Przyciski kontrolne
        btn_container = tk.Frame(p1, bg=THEME["panel"])
        btn_container.pack(fill=tk.X, padx=20, pady=10)

        self.create_button(btn_container, "🗑️ CLEAR", self.czysc).pack(fill=tk.X, pady=5)
        self.create_button(btn_container, "🔄 RECOGNIZE", self.wymus_rozpoznanie).pack(fill=tk.X, pady=5)

        # Statystyki
        stats_frame = tk.Frame(p1, bg=THEME["panel_light"], highlightthickness=1,
                              highlightbackground=THEME["accent"])
        stats_frame.pack(fill=tk.X, padx=20, pady=20)

        tk.Label(stats_frame, text="📊 NETWORK STATS", fg=THEME["accent"],
                bg=THEME["panel_light"], font=("Segoe UI", 10, "bold")).pack(pady=10)

        self.lbl_neurons = tk.Label(stats_frame, text=f"Neurons: {UKRYTE1 + UKRYTE2}",
                                   fg=THEME["text"], bg=THEME["panel_light"], font=("Segoe UI", 9))
        self.lbl_neurons.pack()

        self.lbl_connections = tk.Label(stats_frame, text=f"Connections: {WEJSCIA*UKRYTE1 + UKRYTE1*UKRYTE2:,}",
                                       fg=THEME["text"], bg=THEME["panel_light"], font=("Segoe UI", 9))
        self.lbl_connections.pack(pady=5)

        # ŚRODKOWY PANEL - WIZUALIZACJA
        p2 = tk.Frame(main, bg=THEME["bg"])
        p2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        tk.Label(p2, text="⚡ NEURAL NETWORK VISUALIZATION",
                fg=THEME["accent2"], bg=THEME["bg"],
                font=("Segoe UI", 12, "bold")).pack(pady=10)

        # Canvas wizualizacji z efektem świecenia
        vis_container = tk.Frame(p2, bg=THEME["bg"], highlightthickness=2,
                                highlightbackground=THEME["accent2"])
        vis_container.pack(fill=tk.BOTH, expand=True)

        self.vis_canvas = tk.Canvas(vis_container, bg=THEME["bg"], highlightthickness=0)
        self.vis_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Toggle animacji
        anim_frame = tk.Frame(p2, bg=THEME["bg"])
        anim_frame.pack(pady=10)

        self.var_animacja = tk.BooleanVar(value=True)
        tk.Checkbutton(anim_frame, text="✨ Particle Animation", variable=self.var_animacja,
                      fg=THEME["text"], bg=THEME["bg"], selectcolor=THEME["panel"],
                      activebackground=THEME["bg"], activeforeground=THEME["accent"],
                      font=("Segoe UI", 10, "bold")).pack()

        # PRAWY PANEL - WYNIKI
        p3 = self.create_panel(main, "🎯 RECOGNITION RESULTS", width=380)
        p3.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Wynik rozpoznania
        result_frame = tk.Frame(p3, bg=THEME["panel_light"], highlightthickness=2,
                               highlightbackground=THEME["accent"])
        result_frame.pack(pady=20, padx=20, fill=tk.X)

        tk.Label(result_frame, text="DETECTED:", fg=THEME["text"],
                bg=THEME["panel_light"], font=("Segoe UI", 11)).pack(pady=10)

        self.lbl_wynik = tk.Label(result_frame, text="?",
                                 font=("Segoe UI", 100, "bold"),
                                 fg=THEME["accent"], bg=THEME["panel_light"])
        self.lbl_wynik.pack(pady=10)

        self.lbl_pewnosc = tk.Label(result_frame, text="0.0%",
                                    fg=THEME["accent2"], bg=THEME["panel_light"],
                                    font=("Segoe UI", 16, "bold"))
        self.lbl_pewnosc.pack(pady=10)

        # Progress bar pewności
        self.confidence_bar = ttk.Progressbar(result_frame, orient="horizontal",
                                             length=300, mode="determinate")
        self.confidence_bar.pack(pady=10)

        # Top 5 predykcji
        tk.Label(p3, text="📈 TOP 5 PREDICTIONS", fg=THEME["accent"],
                bg=THEME["panel"], font=("Segoe UI", 10, "bold")).pack(pady=15)

        self.top5_frame = tk.Frame(p3, bg=THEME["panel"])
        self.top5_frame.pack(fill=tk.X, padx=20)

        self.top5_labels = []
        for i in range(5):
            frame = tk.Frame(self.top5_frame, bg=THEME["panel_light"])
            frame.pack(fill=tk.X, pady=3)

            lbl = tk.Label(frame, text=f"{i+1}. ? - 0%", fg=THEME["text"],
                          bg=THEME["panel_light"], font=("Segoe UI", 10), anchor="w")
            lbl.pack(side=tk.LEFT, padx=10, pady=5)
            self.top5_labels.append(lbl)

        # Separator
        tk.Frame(p3, height=2, bg=THEME["accent"]).pack(fill=tk.X, pady=20, padx=20)

        # TRENING
        tk.Label(p3, text="🚀 TRAINING", fg=THEME["accent2"],
                bg=THEME["panel"], font=("Segoe UI", 11, "bold")).pack(pady=10)

        self.create_button(p3, "📊 LOAD & TRAIN EMNIST",
                         self.start_watek_emnist).pack(fill=tk.X, padx=20, pady=10)

        self.progress = ttk.Progressbar(p3, orient="horizontal", mode="determinate")
        self.progress.pack(fill=tk.X, padx=20, pady=10)

        self.lbl_status = tk.Label(p3, text="Ready to train...",
                                  fg=THEME["text"], bg=THEME["panel"],
                                  wraplength=340, font=("Segoe UI", 9))
        self.lbl_status.pack(pady=5)

    def create_panel(self, parent, title, width=None):
        """Tworzy stylowy panel z tytułem"""
        frame = tk.Frame(parent, bg=THEME["panel"], width=width,
                        highlightthickness=2, highlightbackground=THEME["accent2"])
        if width:
            frame.pack_propagate(False)

        tk.Label(frame, text=title, fg=THEME["accent"], bg=THEME["panel"],
                font=("Segoe UI", 12, "bold")).pack(pady=15)

        return frame

    def create_button(self, parent, text, command, width=None):
        """Tworzy stylowy przycisk z efektem hover"""
        btn = tk.Button(parent, text=text, command=command,
                       bg=THEME["button_bg"], fg="white",
                       font=("Segoe UI", 10, "bold"),
                       relief="flat", cursor="hand2", pady=12,
                       activebackground=THEME["button_hover"],
                       activeforeground="white")
        if width:
            btn.config(width=width)

        # Efekt hover
        btn.bind("<Enter>", lambda e: btn.config(bg=THEME["button_hover"]))
        btn.bind("<Leave>", lambda e: btn.config(bg=THEME["button_bg"]))

        return btn

    def inicjuj_wizualizacje(self):
        """Inicjalizuje pozycje neuronów dla wizualizacji"""
        self.root.update()
        w = self.vis_canvas.winfo_width()
        h = self.vis_canvas.winfo_height()

        # Pozycje warstw - bardziej estetyczny układ
        margin = 80

        # Warstwa wejściowa (próbka neuronów)
        self.k_input = [(margin, h * (i+1)/(21)) for i in range(20)]

        # Warstwa ukryta 1 (próbka)
        x1 = w * 0.35
        self.k_hidden1 = [(x1, h * (i+1)/(33)) for i in range(32)]

        # Warstwa ukryta 2 (próbka)
        x2 = w * 0.6
        self.k_hidden2 = [(x2, h * (i+1)/(25)) for i in range(24)]

        # Warstwa wyjściowa
        x3 = w - margin
        self.k_output = [(x3, h * (i+1)/(WYJSCIA+1)) for i in range(WYJSCIA)]

        # Rysuj strukturę sieci
        self.rysuj_strukture_sieci()

        # Inicjalizuj animator
        self.animator = AnimowanaWizualizacja(self.vis_canvas, self.siec)

    def rysuj_strukture_sieci(self):
        """Rysuje statyczną strukturę sieci"""
        # Neurony wejściowe
        for x, y in self.k_input:
            self.vis_canvas.create_oval(x-4, y-4, x+4, y+4,
                                       fill=THEME["neuron_inactive"],
                                       outline=THEME["accent"], tags="neuron_in")

        # Neurony ukryte 1
        for i, (x, y) in enumerate(self.k_hidden1):
            self.vis_canvas.create_oval(x-5, y-5, x+5, y+5,
                                       fill=THEME["neuron_inactive"],
                                       outline="", tags=f"h1_{i}")

        # Neurony ukryte 2
        for i, (x, y) in enumerate(self.k_hidden2):
            self.vis_canvas.create_oval(x-5, y-5, x+5, y+5,
                                       fill=THEME["neuron_inactive"],
                                       outline="", tags=f"h2_{i}")

        # Neurony wyjściowe z etykietami
        for i, (x, y) in enumerate(self.k_output):
            self.vis_canvas.create_oval(x-6, y-6, x+6, y+6,
                                       fill=THEME["neuron_inactive"],
                                       outline="", tags=f"out_{i}")
            self.vis_canvas.create_text(x+25, y, text=ZNAKI[i],
                                       fill=THEME["text"],
                                       font=("Courier", 10, "bold"),
                                       tags=f"label_{i}")

    def petla_animacji(self):
        """Główna pętla animacji - aktualizuje wizualizację"""
        if self.var_animacja.get():
            self.aktualizuj_wizualizacje_zaawansowana()

        self.root.after(50, self.petla_animacji)  # 20 FPS

    def aktualizuj_wizualizacje_zaawansowana(self):
        """Zaawansowana wizualizacja z animacjami i efektami"""
        # Usuń stare linie i cząsteczki
        self.vis_canvas.delete("connection")
        self.vis_canvas.delete("particle")

        # Pobierz aktywacje
        h1_vals = self.siec.ukryta1[0]
        h2_vals = self.siec.ukryta2[0]
        out_vals = self.siec.wyjscie[0]

        # Wybierz najbardziej aktywne neurony
        top_h1 = np.argsort(h1_vals)[-32:]
        top_h2 = np.argsort(h2_vals)[-24:]
        best_out = np.argmax(out_vals)

        # Aktualizuj kolory neuronów ukrytych 1
        for i, idx in enumerate(top_h1):
            if idx < len(h1_vals):
                val = h1_vals[idx]
                intensity = int(min(255, val * 255))
                color = f"#{0:02x}{intensity:02x}{intensity:02x}"

                if i < len(self.k_hidden1):
                    self.vis_canvas.itemconfig(f"h1_{i}", fill=color)

                    # Dodaj cząsteczki płynące do warstwy 2
                    if self.var_animacja.get() and val > 0.5 and np.random.random() > 0.7:
                        x1, y1 = self.k_hidden1[i]
                        if len(top_h2) > 0:
                            target_idx = np.random.choice(len(self.k_hidden2))
                            x2, y2 = self.k_hidden2[target_idx]
                            self.animator.dodaj_particle(x1, y1, x2, y2, THEME["synapse_pos"])

        # Aktualizuj kolory neuronów ukrytych 2
        for i, idx in enumerate(top_h2):
            if idx < len(h2_vals):
                val = h2_vals[idx]
                intensity = int(min(255, val * 400))  # Zwiększona intensywność
                color = f"#{intensity:02x}{0:02x}{intensity:02x}"

                if i < len(self.k_hidden2):
                    self.vis_canvas.itemconfig(f"h2_{i}", fill=color)

                    # Rysuj połączenia do zwycięskiego neuronu wyjściowego
                    x1, y1 = self.k_hidden2[i]
                    x2, y2 = self.k_output[best_out]

                    # Pobierz wagę połączenia
                    if idx < self.siec.w3.shape[0]:
                        waga = self.siec.w3[idx, best_out]
                        if abs(waga) > 0.1:
                            width = min(3, abs(waga) * 1.5)
                            color = THEME["synapse_pos"] if waga > 0 else THEME["synapse_neg"]
                            alpha = int(min(100, abs(waga) * 50))

                            self.vis_canvas.create_line(x1, y1, x2, y2,
                                                       fill=color, width=width,
                                                       tags="connection")

                    # Dodaj cząsteczki do outputu
                    if self.var_animacja.get() and val > 0.3 and np.random.random() > 0.8:
                        self.animator.dodaj_particle(x1, y1, x2, y2, THEME["accent"])

        # Aktualizuj neurony wyjściowe
        top5_indices = np.argsort(out_vals)[-5:][::-1]
        for i in range(WYJSCIA):
            val = out_vals[i]

            if i == best_out:
                # Zwycięzca - pulsujący efekt
                pulse = 0.3 * np.sin(time.time() * 5) + 0.7
                self.vis_canvas.itemconfig(f"out_{i}",
                                          fill=THEME["neuron_active"],
                                          outline=THEME["accent"],
                                          width=2)
                self.vis_canvas.itemconfig(f"label_{i}",
                                          fill="white",
                                          font=("Courier", 12, "bold"))
            elif i in top5_indices:
                # Top 5
                intensity = int(val * 200)
                color = f"#{intensity:02x}{intensity:02x}{0:02x}"
                self.vis_canvas.itemconfig(f"out_{i}", fill=color, outline="")
                self.vis_canvas.itemconfig(f"label_{i}",
                                          fill=THEME["text"],
                                          font=("Courier", 10, "bold"))
            else:
                # Pozostałe
                self.vis_canvas.itemconfig(f"out_{i}",
                                          fill=THEME["neuron_inactive"],
                                          outline="")
                self.vis_canvas.itemconfig(f"label_{i}",
                                          fill="#555",
                                          font=("Courier", 9))

        # Aktualizuj cząsteczki
        if self.var_animacja.get():
            self.animator.aktualizuj_particles()

    def rozpocznij_rysowanie(self, event):
        """Rozpoczyna rysowanie"""
        self.rysowanie_aktywne = True

    def rysuj(self, event):
        """Obsługa rysowania na canvasie"""
        if not self.rysowanie_aktywne:
            return

        x, y = event.x, event.y
        r = 18  # Rozmiar pędzla

        # Rysuj na canvasie
        self.canvas.create_oval(x-r, y-r, x+r, y+r,
                               fill="white", outline="")

        # Aktualizuj tablicę 28x28
        scale = 320 / ROZMIAR
        grid_x = int(x / scale)
        grid_y = int(y / scale)

        # Rysuj także sąsiednie piksele dla gładszości
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                gx, gy = grid_x + dx, grid_y + dy
                if 0 <= gx < ROZMIAR and 0 <= gy < ROZMIAR:
                    dist = np.sqrt(dx**2 + dy**2)
                    val = max(0, 1.0 - dist * 0.4)
                    self.plansza[gy, gx] = min(1.0, self.plansza[gy, gx] + val)

    def koniec_rysowania(self, event):
        """Kończy rysowanie i rozpoznaje znak"""
        self.rysowanie_aktywne = False
        self.rozpoznaj()

    def wymus_rozpoznanie(self):
        """Wymusza rozpoznanie bez rysowania"""
        self.rozpoznaj()

    def rozpoznaj(self):
        """Rozpoznaje narysowany znak"""
        # Normalizuj dane wejściowe
        x = self.plansza.flatten().reshape(1, -1)

        # Forward pass z zapisem historii
        wynik = self.siec.forward(x, record_history=True)

        # Znajdź najlepsze dopasowanie
        idx = np.argmax(wynik[0])
        pewnosc = wynik[0][idx] * 100

        # Aktualizuj UI
        self.lbl_wynik.config(text=ZNAKI[idx])
        self.lbl_pewnosc.config(text=f"{pewnosc:.1f}%")
        self.confidence_bar['value'] = pewnosc

        # Aktualizuj top 5
        top5_idx = np.argsort(wynik[0])[-5:][::-1]
        for i, idx in enumerate(top5_idx):
            conf = wynik[0][idx] * 100
            self.top5_labels[i].config(
                text=f"{i+1}. {ZNAKI[idx]} - {conf:.1f}%"
            )

        # Animacja wizualizacji zostanie zaktualizowana w pętli

    def czysc(self):
        """Czyści canvas"""
        self.canvas.delete("all")
        self.plansza = np.zeros((ROZMIAR, ROZMIAR))
        self.lbl_wynik.config(text="?")
        self.lbl_pewnosc.config(text="0.0%")
        self.confidence_bar['value'] = 0

        for lbl in self.top5_labels:
            lbl.config(text="")

    def zapisz_model(self):
        """Zapisuje wytrenowany model"""
        self.siec.save_model("model_ocr.pkl")
        messagebox.showinfo("Success", "Model został zapisany jako model_ocr.pkl")

    def wczytaj_model(self):
        """Wczytuje zapisany model"""
        filepath = filedialog.askopenfilename(
            title="Wybierz model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filepath and self.siec.load_model(filepath):
            messagebox.showinfo("Success", "Model został wczytany!")
        else:
            messagebox.showerror("Error", "Nie udało się wczytać modelu")

    def start_watek_emnist(self):
        """Uruchamia trening w osobnym wątku"""
        plik = filedialog.askopenfilename(
            title="Wybierz EMNIST CSV",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if not plik:
            return

        t = threading.Thread(target=self.proces_emnist, args=(plik,), daemon=True)
        t.start()

    def proces_emnist(self, plik):
        """Proces treningu na EMNIST z walidacją"""
        if not PANDAS_DOSTEPNY:
            self.lbl_status.config(text="Błąd: Brak biblioteki pandas!")
            return

        try:
            self.lbl_status.config(text="📥 Wczytywanie danych...", fg="yellow")

            # Wczytaj dane
            df = pd.read_csv(plik, header=None)
            labels = df.iloc[:, 0].values
            pixels = df.iloc[:, 1:].values / 255.0  # Normalizacja

            # Filtruj tylko 0-35 (cyfry i litery)
            mask = labels < 36
            labels = labels[mask]
            pixels = pixels[mask]

            # Tasowanie
            perm = np.random.permutation(len(labels))
            labels = labels[perm]
            pixels = pixels[perm]

            # Split train/validation (90/10)
            split_idx = int(len(labels) * 0.9)
            train_x, val_x = pixels[:split_idx], pixels[split_idx:]
            train_y_idx, val_y_idx = labels[:split_idx], labels[split_idx:]

            # One-hot encoding
            train_y = np.zeros((len(train_y_idx), WYJSCIA))
            val_y = np.zeros((len(val_y_idx), WYJSCIA))
            train_y[np.arange(len(train_y_idx)), train_y_idx] = 1
            val_y[np.arange(len(val_y_idx)), val_y_idx] = 1

            # Trening
            batch_size = 32
            epochs = 5
            total_batches = (len(train_x) // batch_size) * epochs
            batch_count = 0

            self.lbl_status.config(text=f"🚀 Trening... Próbek: {len(train_x)}")

            for epoch in range(epochs):
                epoch_loss = 0
                num_batches = 0

                # Shuffle co epokę
                perm = np.random.permutation(len(train_x))
                train_x_shuffled = train_x[perm]
                train_y_shuffled = train_y[perm]

                for i in range(0, len(train_x), batch_size):
                    batch_x = train_x_shuffled[i:i+batch_size]
                    batch_y = train_y_shuffled[i:i+batch_size]

                    # Adaptive learning rate
                    lr = 0.01 * (0.95 ** epoch)
                    loss = self.siec.train(batch_x, batch_y, lr=lr)

                    epoch_loss += loss
                    num_batches += 1
                    batch_count += 1

                    # Aktualizuj progress
                    progress = (batch_count / total_batches) * 100
                    self.progress['value'] = progress

                    if num_batches % 50 == 0:
                        avg_loss = epoch_loss / num_batches
                        self.lbl_status.config(
                            text=f"Epoch {epoch+1}/{epochs} | Batch {num_batches} | Loss: {avg_loss:.4f}"
                        )
                        self.root.update()

                # Walidacja po każdej epoce
                val_predictions = self.siec.forward(val_x)
                val_pred_labels = np.argmax(val_predictions, axis=1)
                val_accuracy = np.mean(val_pred_labels == val_y_idx) * 100

                avg_loss = epoch_loss / num_batches
                self.lbl_status.config(
                    text=f"✓ Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_accuracy:.2f}%",
                    fg=THEME["synapse_pos"]
                )
                self.root.update()

                time.sleep(0.5)

            # Końcowa walidacja
            final_predictions = self.siec.forward(val_x)
            final_pred_labels = np.argmax(final_predictions, axis=1)
            final_accuracy = np.mean(final_pred_labels == val_y_idx) * 100

            self.lbl_status.config(
                text=f"✅ GOTOWE! Final Accuracy: {final_accuracy:.2f}%",
                fg=THEME["synapse_pos"]
            )
            self.progress['value'] = 100

            # Auto-save
            self.siec.save_model("model_ocr.pkl")

            messagebox.showinfo(
                "Trening zakończony!",
                f"Dokładność walidacyjna: {final_accuracy:.2f}%\n"
                f"Model zapisany jako model_ocr.pkl"
            )

        except Exception as e:
            self.lbl_status.config(text=f"❌ Błąd: {str(e)}", fg=THEME["synapse_neg"])
            messagebox.showerror("Błąd", f"Błąd podczas treningu:\n{str(e)}")


def main():
    """Główna funkcja uruchamiająca aplikację"""
    root = tk.Tk()
    app = AplikacjaNeural(root)
    root.mainloop()


if __name__ == "__main__":
    main()
