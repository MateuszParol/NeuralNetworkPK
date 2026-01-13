import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import threading

# Sprawdzamy czy mamy pandas
try:
    import pandas as pd
    PANDAS_DOSTEPNY = True
except ImportError:
    PANDAS_DOSTEPNY = False
    print("BRAK PANDAS! Zainstaluj: pip install pandas")

# --- KONFIGURACJA ---
THEME = {
    "bg": "#121212", "panel": "#1E1E1E", "accent": "#007acc", 
    "text": "#e0e0e0", "synapse": "#00E676", "alert": "#FF5252"
}

ROZMIAR = 28
WEJSCIA = ROZMIAR * ROZMIAR
UKRYTE = 140 # Zwiększyłem lekko liczbę neuronów dla lepszej pamięci liter
WYJSCIA = 36 
ZNAKI = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

class SiecNeuronowa:
    def __init__(self):
        np.random.seed(42)
        self.w1 = np.random.randn(WEJSCIA, UKRYTE) * np.sqrt(2/WEJSCIA)
        self.w2 = np.random.randn(UKRYTE, WYJSCIA) * np.sqrt(2/UKRYTE)
        self.b1 = np.zeros((1, UKRYTE))
        self.b2 = np.zeros((1, WYJSCIA))
        
        self.ukryta = np.zeros((1, UKRYTE))
        self.wyjscie = np.zeros((1, WYJSCIA))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def pochodna_sigmoid(self, x):
        return x * (1 - x)

    def forward(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        self.ukryta = self.sigmoid(z1)
        z2 = np.dot(self.ukryta, self.w2) + self.b2
        self.wyjscie = self.sigmoid(z2)
        return self.wyjscie

    def train(self, x, y, lr=0.1, limit=5.0):
        wyjscie = self.forward(x)
        
        blad = y - wyjscie
        d_out = blad * self.pochodna_sigmoid(wyjscie)
        
        blad_hid = d_out.dot(self.w2.T)
        d_hid = blad_hid * self.pochodna_sigmoid(self.ukryta)
        
        self.w2 += self.ukryta.T.dot(d_out) * lr
        self.b2 += np.sum(d_out, axis=0, keepdims=True) * lr
        self.w1 += x.T.dot(d_hid) * lr
        self.b1 += np.sum(d_hid, axis=0, keepdims=True) * lr
        
        self.w1 = np.clip(self.w1, -limit, limit)
        self.w2 = np.clip(self.w2, -limit, limit)

class AplikacjaV8:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network v8.0 (Shuffled EMNIST)")
        self.root.geometry("1280x800")
        self.root.configure(bg=THEME["bg"])
        
        self.siec = SiecNeuronowa()
        self.plansza = np.zeros((ROZMIAR, ROZMIAR))
        
        self.buduj_interfejs()
        self.inicjuj_wizualizacje()

    def buduj_interfejs(self):
        main = tk.Frame(self.root, bg=THEME["bg"])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # LEWY PANEL (RYSOWANIE)
        p1 = tk.Frame(main, bg=THEME["panel"], width=320)
        p1.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        p1.pack_propagate(False)
        
        tk.Label(p1, text="RYSOWANIE", fg=THEME["text"], bg=THEME["panel"], font=("Segoe UI", 12, "bold")).pack(pady=15)
        
        self.canvas = tk.Canvas(p1, width=280, height=280, bg="black", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.rysuj)
        self.canvas.bind("<ButtonRelease-1>", self.koniec_rysowania)
        self.canvas.bind("<Button-3>", lambda e: self.czysc())

        tk.Button(p1, text="RESET (PRAWY PRZYCISK)", command=self.czysc, bg="#D32F2F", fg="white", 
                  relief="flat", font=("Segoe UI", 10, "bold"), pady=8).pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(p1, text="LIMITER WAG (Max Value)", fg="gray", bg=THEME["panel"]).pack(pady=(30,5))
        self.var_limit = tk.DoubleVar(value=2.0)
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TScale", background=THEME["panel"])
        ttk.Scale(p1, from_=0.1, to=10.0, variable=self.var_limit).pack(fill=tk.X, padx=20)

        # ŚRODKOWY PANEL (WIZUALIZACJA)
        p2 = tk.Frame(main, bg=THEME["bg"])
        p2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        tk.Label(p2, text="NEURAL PATHWAYS (LIVE)", fg="#555", bg=THEME["bg"], font=("Segoe UI", 10)).pack(pady=5)
        self.vis_canvas = tk.Canvas(p2, bg=THEME["bg"], highlightthickness=0)
        self.vis_canvas.pack(fill=tk.BOTH, expand=True)

        # PRAWY PANEL (WYNIKI I TRENING)
        p3 = tk.Frame(main, bg=THEME["panel"], width=320)
        p3.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        p3.pack_propagate(False)
        
        tk.Label(p3, text="ROZPOZNANO:", fg=THEME["text"], bg=THEME["panel"], font=("Segoe UI", 12)).pack(pady=15)
        self.lbl_wynik = tk.Label(p3, text="?", font=("Segoe UI", 80, "bold"), fg=THEME["accent"], bg=THEME["panel"])
        self.lbl_wynik.pack()
        self.lbl_pewnosc = tk.Label(p3, text="0%", fg="gray", bg=THEME["panel"], font=("Segoe UI", 12))
        self.lbl_pewnosc.pack()
        
        tk.Frame(p3, height=1, bg="#444").pack(fill=tk.X, pady=30, padx=20)
        
        # EMNIST
        tk.Label(p3, text="TRENING (CSV)", fg=THEME["text"], bg=THEME["panel"], font=("Segoe UI", 11, "bold")).pack(pady=5)
        tk.Label(p3, text="Wymaga: emnist-balanced-train.csv", fg="gray", bg=THEME["panel"], font=("Segoe UI", 8)).pack()
        
        self.btn_load = tk.Button(p3, text="WCZYTAJ I TRENUJ (MIX)", command=self.start_watek_emnist, 
                                  bg="#0984e3", fg="white", font=("Segoe UI", 10, "bold"), pady=12, relief="flat", cursor="hand2")
        self.btn_load.pack(fill=tk.X, padx=20, pady=15)
        
        self.progress = ttk.Progressbar(p3, orient="horizontal", length=200, mode="determinate")
        self.progress.pack(fill=tk.X, padx=20, pady=5)
        
        self.lbl_status = tk.Label(p3, text="Oczekiwanie na dane...", fg="gray", bg=THEME["panel"], wraplength=280)
        self.lbl_status.pack(pady=5)

    def inicjuj_wizualizacje(self):
        self.root.update()
        w = self.vis_canvas.winfo_width()
        h = self.vis_canvas.winfo_height()
        
        self.k_in = [(30, (h/22)*(i+1)+20) for i in range(20)]
        self.k_hid = [((w/2)-40 + (i//16)*20, (h/18)*(i%16+1)) for i in range(UKRYTE)]
        self.k_out = [(w-50, (h/38)*(i+1)) for i in range(WYJSCIA)]
        
        for x,y in self.k_in: self.vis_canvas.create_oval(x-2, y-2, x+2, y+2, fill="#444", outline="")
        for i, (x,y) in enumerate(self.k_hid): self.vis_canvas.create_oval(x-3, y-3, x+3, y+3, fill="#222", outline="", tags=f"h{i}")
        for i, (x,y) in enumerate(self.k_out):
            self.vis_canvas.create_text(x+20, y, text=ZNAKI[i], fill="#555", font=("Arial", 9, "bold"), tags=f"t{i}")
            self.vis_canvas.create_oval(x-5, y-5, x+5, y+5, fill="#222", outline="", tags=f"o{i}")

    def aktualizuj_wizualizacje(self):
        self.vis_canvas.delete("line")
        limit = self.var_limit.get()
        
        idx_hid = np.argsort(self.siec.ukryta[0])[-25:] # Pokaż top 25 aktywnych neuronów
        idx_win = np.argmax(self.siec.wyjscie[0])
        
        for i in idx_hid:
            val = self.siec.ukryta[0][i]
            col_hid = f"#{0:02x}{int(val*255):02x}{0:02x}"
            self.vis_canvas.itemconfig(f"h{i}", fill=col_hid)
            
            waga = self.siec.w2[i][idx_win]
            width = abs(waga) * 2 / limit
            col_line = THEME["synapse"] if waga > 0 else THEME["alert"]
            
            # Rysuj tylko silne połączenia
            if width > 0.3:
                x1, y1 = self.k_hid[i]
                x2, y2 = self.k_out[idx_win]
                self.vis_canvas.create_line(x1, y1, x2, y2, fill=col_line, width=width, tags="line")

        for i in range(WYJSCIA):
            val = self.siec.wyjscie[0][i]
            is_win = (i == idx_win)
            self.vis_canvas.itemconfig(f"o{i}", fill=THEME["accent"] if is_win else "#222")
            self.vis_canvas.itemconfig(f"t{i}", fill="white" if is_win else "#444")

    # --- WĄTEK EMNIST Z TASOWANIEM (SHUFFLE) ---
    def start_watek_emnist(self):
        plik = filedialog.askopenfilename(title="Wybierz CSV (emnist-balanced)", filetypes=[("CSV", "*.csv")])
        if not plik: return
        t = threading.Thread(target=self.proces_emnist, args=(plik,))
        t.start()

    def proces_emnist(self, plik):
        if not PANDAS_DOSTEPNY: return

        try:
            self.lbl_status.config(text="Wczytywanie i tasowanie...", fg="yellow")
            self.btn_load.config(state="disabled")
            
            # 1. Wczytanie
            df = pd.read_csv(plik, header=None)
            labels = df.iloc[:, 0].values
            pixels = df.iloc[:, 1:].values
            
            # 2. Filtrowanie (tylko 0-35)
            mask = labels < 36
            labels = labels[mask]
            pixels = pixels[mask]
            
            # 3. TASOWANIE (KLUCZOWA POPRAWKA!)
            # Mieszamy dane, żeby sieć widziała na zmianę cyfry i litery
            perm = np.random.permutation(len(labels))
            labels = labels[perm]
            pixels = pixels[perm]
            
            # Bierzemy więcej próbek,