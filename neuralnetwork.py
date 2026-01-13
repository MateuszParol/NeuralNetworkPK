import numpy as np
import tkinter as tk
from tkinter import messagebox
import urllib.request
import os

# --- KONFIGURACJA INŻYNIERSKA ---
ROZMIAR_SIATKI = 28  # Standard MNIST
WEJSCIA = ROZMIAR_SIATKI * ROZMIAR_SIATKI  # 784
UKRYTE = 64          # Warstwa ukryta (zmniejszona lekko dla czytelności wizualizacji)
WYJSCIA = 36         # 0-9 + A-Z
ZNAKI = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Kolorystyka "Professional Dark"
COL_BG = "#1e1e1e"       # Tło główne
COL_PANEL = "#252526"    # Tło paneli
COL_ACCENT = "#007acc"   # Niebieski akcent (VS Code style)
COL_TEXT = "#cccccc"     # Szary tekst
COL_NEURON_OFF = "#333333"
COL_NEURON_ON = "#00ff00"

class SiecNeuronowa:
    def __init__(self):
        np.random.seed(42)
        # Inicjalizacja He (Dla Sigmoida/ReLU)
        self.w1 = np.random.randn(WEJSCIA, UKRYTE) * np.sqrt(2/WEJSCIA)
        self.b1 = np.zeros((1, UKRYTE))
        self.w2 = np.random.randn(UKRYTE, WYJSCIA) * np.sqrt(2/UKRYTE)
        self.b2 = np.zeros((1, WYJSCIA))
        
        # Pamięć stanów do wizualizacji
        self.ukryta_aktywacja = np.zeros((1, UKRYTE))
        self.wyjscie_aktywacja = np.zeros((1, WYJSCIA))

    def sigmoid(self, x):
        # Zabezpieczenie numeryczne
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def pochodna_sigmoid(self, x):
        return x * (1 - x)

    def forward(self, x):
        # Warstwa 1
        z1 = np.dot(x, self.w1) + self.b1
        self.ukryta_aktywacja = self.sigmoid(z1)
        
        # Warstwa 2
        z2 = np.dot(self.ukryta_aktywacja, self.w2) + self.b2
        self.wyjscie_aktywacja = self.sigmoid(z2)
        return self.wyjscie_aktywacja

    def backward(self, x, y, lr=0.1):
        # Propagacja w przód (musimy mieć świeży stan)
        wyjscie = self.forward(x)
        
        # Obliczenie błędu
        blad = y - wyjscie
        
        # Propagacja wsteczna
        d_wyjscie = blad * self.pochodna_sigmoid(wyjscie)
        blad_ukryty = d_wyjscie.dot(self.w2.T)
        d_ukryty = blad_ukryty * self.pochodna_sigmoid(self.ukryta_aktywacja)

        # Aktualizacja wag
        self.w2 += self.ukryta_aktywacja.T.dot(d_wyjscie) * lr
        self.b2 += np.sum(d_wyjscie, axis=0, keepdims=True) * lr
        self.w1 += x.T.dot(d_ukryty) * lr
        self.b1 += np.sum(d_ukryty, axis=0, keepdims=True) * lr

        # --- LIMITER WAG (Weight Clipping) ---
        # Tutaj znajduje się mechanizm zapobiegający "wybuchom" wag
        limit = 5.0
        self.w1 = np.clip(self.w1, -limit, limit)
        self.w2 = np.clip(self.w2, -limit, limit)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Studio v4.0")
        self.root.geometry("1200x700")
        self.root.configure(bg=COL_BG)
        
        self.siec = SiecNeuronowa()
        self.plansza = np.zeros((ROZMIAR_SIATKI, ROZMIAR_SIATKI))
        
        self._buduj_interfejs()
        self._inicjuj_wizualizacje_sieci()

    def _buduj_interfejs(self):
        # Główny kontener
        main_frame = tk.Frame(self.root, bg=COL_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- KOLUMNA 1: RYSOWANIE (INPUT) ---
        col1 = tk.Frame(main_frame, bg=COL_PANEL, width=300)
        col1.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        tk.Label(col1, text="INPUT (28x28)", fg=COL_TEXT, bg=COL_PANEL, font=("Consolas", 12)).pack(pady=10)
        
        self.canvas = tk.Canvas(col1, width=280, height=280, bg="black", highlightthickness=0)
        self.canvas.pack(pady=10, padx=10)
        self.canvas.bind("<B1-Motion>", self._rysuj)
        self.canvas.bind("<ButtonRelease-1>", self._koniec_rysowania)
        self.canvas.bind("<Button-3>", lambda e: self._reset())

        tk.Button(col1, text="RESET [Prawy Przycisk]", command=self._reset, 
                  bg="#d63031", fg="white", relief="flat", pady=5).pack(fill=tk.X, padx=10)

        # --- KOLUMNA 2: WIZUALIZACJA SIECI (HIDDEN LAYER) ---
        col2 = tk.Frame(main_frame, bg=COL_PANEL)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(col2, text="ARCHITEKTURA SIECI (LIVE)", fg=COL_TEXT, bg=COL_PANEL, font=("Consolas", 12)).pack(pady=10)
        
        # Canvas do rysowania neuronów i połączeń
        self.viz_canvas = tk.Canvas(col2, bg="#111111", highlightthickness=0)
        self.viz_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- KOLUMNA 3: STEROWANIE I WYNIKI (OUTPUT) ---
        col3 = tk.Frame(main_frame, bg=COL_PANEL, width=300)
        col3.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        tk.Label(col3, text="STEROWANIE / WYNIKI", fg=COL_TEXT, bg=COL_PANEL, font=("Consolas", 12)).pack(pady=10)
        
        # Wyświetlacz wyniku
        self.lbl_wynik = tk.Label(col3, text="?", font=("Arial", 48, "bold"), fg=COL_ACCENT, bg=COL_PANEL)
        self.lbl_wynik.pack(pady=20)
        self.lbl_detale = tk.Label(col3, text="Oczekiwanie...", fg=COL_TEXT, bg=COL_PANEL)
        self.lbl_detale.pack()

        # Przyciski
        tk.Button(col3, text="ZGADNIJ", command=self._zgadnij, 
                  bg=COL_ACCENT, fg="white", font=("Arial", 12, "bold"), relief="flat", pady=10).pack(fill=tk.X, padx=10, pady=20)

        tk.Label(col3, text="TRENING", fg=COL_TEXT, bg=COL_PANEL).pack(pady=5)
        
        self.btn_mnist = tk.Button(col3, text="Pobierz i Trenuj MNIST (0-9)", command=self._start_mnist, 
                  bg="#f0932b", fg="black", relief="flat")
        self.btn_mnist.pack(fill=tk.X, padx=10, pady=5)
        
        # Lista rozwijana do ręcznego uczenia
        self.var_znak = tk.StringVar(value="A")
        frame_reczny = tk.Frame(col3, bg=COL_PANEL)
        frame_reczny.pack(fill=tk.X, padx=10, pady=10)
        
        tk.OptionMenu(frame_reczny, self.var_znak, *ZNAKI).pack(side=tk.LEFT)
        tk.Button(frame_reczny, text="Naucz", command=self._ucz_recznie, bg="#6ab04c", fg="white", relief="flat").pack(side=tk.RIGHT, fill=tk.X, expand=True)

        self.lbl_status = tk.Label(col3, text="Gotowy", fg="gray", bg=COL_PANEL, font=("Arial", 8))
        self.lbl_status.pack(side=tk.BOTTOM, pady=10)

    def _inicjuj_wizualizacje_sieci(self):
        # Rysuje statyczną strukturę sieci (kółka), które potem będziemy podświetlać
        self.viz_canvas.update()
        w = self.viz_canvas.winfo_width()
        h = self.viz_canvas.winfo_height()
        
        self.neuron_coords = [] # Lista współrzędnych do animacji
        
        # Warstwa wejściowa (symboliczna - lewa strona)
        x_in = 50
        for i in range(10): # Rysujemy tylko 10 symbolicznych wejść
            y = (h / 12) * (i + 1)
            self.viz_canvas.create_oval(x_in-5, y-5, x_in+5, y+5, fill=COL_NEURON_OFF, outline="")
        
        # Warstwa ukryta (środek)
        x_hid = w / 2
        # Rysujemy reprezentację 64 neuronów w siatce (np. 4 kolumny po 16)
        self.hidden_ids = []
        for i in range(UKRYTE):
            row = i % 16
            col = i // 16
            xx = x_hid - 30 + (col * 20)
            yy = 50 + (row * (h-100)/16)
            nid = self.viz_canvas.create_oval(xx-4, yy-4, xx+4, yy+4, fill=COL_NEURON_OFF, outline="gray")
            self.hidden_ids.append(nid)

        # Warstwa wyjściowa (prawa strona)
        x_out = w - 50
        self.output_ids = []
        for i in range(WYJSCIA):
            # Rysujemy w 2 kolumnach żeby się zmieściło
            col = i % 2
            row = i // 2
            xx = x_out + (col * 20)
            yy = 20 + (row * (h-40)/18)
            
            # Kółko
            nid = self.viz_canvas.create_oval(xx-6, yy-6, xx+6, yy+6, fill=COL_NEURON_OFF, outline="")
            self.output_ids.append(nid)
            # Etykieta (0, 1, A, B...)
            self.viz_canvas.create_text(xx+15, yy, text=ZNAKI[i], fill="gray", font=("Arial", 8))

        # Linie połączeń (symboliczne) - rysujemy je pod spodem
        self.viz_canvas.tag_lower("all")

    def _aktualizuj_wizualizacje_live(self):
        # Pobieramy stany aktywacji z mózgu
        hidden_acts = self.siec.ukryta_aktywacja[0] # wektor 64 liczb
        output_acts = self.siec.wyjscie_aktywacja[0] # wektor 36 liczb

        # 1. Aktualizacja Warstwy Ukrytej
        for i, val in enumerate(hidden_acts):
            # Normalizacja koloru (0.0 -> czarny, 1.0 -> jasny zielony)
            intensity = int(val * 255)
            hex_col = f"#{0:02x}{intensity:02x}{0:02x}"
            self.viz_canvas.itemconfig(self.hidden_ids[i], fill=hex_col)

        # 2. Aktualizacja Warstwy Wyjściowej
        for i, val in enumerate(output_acts):
            intensity = int(val * 255)
            # Jeśli neuron jest bardzo aktywny (>0.7), robimy go niebieskim
            if val > 0.7:
                hex_col = "#00a8ff"
                outline = "white"
            else:
                hex_col = f"#{intensity:02x}{intensity:02x}{intensity:02x}" # odcienie szarości dla nieaktywnych
                outline = ""
            
            self.viz_canvas.itemconfig(self.output_ids[i], fill=hex_col, outline=outline)

    def _rysuj(self, event):
        x, y = event.x, event.y
        r = 12 # Promień pędzla wizualnego
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        
        # Matematyka (Rysowanie do macierzy)
        scale = 280 / ROZMIAR_SIATKI
        grid_x, grid_y = int(x / scale), int(y / scale)
        
        # Pędzel "rozmyty" (dodaje wartości do sąsiadów)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < ROZMIAR_SIATKI and 0 <= ny < ROZMIAR_SIATKI:
                    self.plansza[ny][nx] = min(self.plansza[ny][nx] + 0.6, 1.0)
        
        # Przy każdym ruchu pędzla odświeżamy wizualizację sieci!
        # Robimy "szybki forward" bez treningu, żeby zobaczyć jak sieć reaguje na żywo
        flat_data = self.plansza.flatten().reshape(1, WEJSCIA)
        self.siec.forward(flat_data)
        self._aktualizuj_wizualizacje_live()

    def _koniec_rysowania(self, event):
        # Tylko wyzwala zgadywanie po puszczeniu myszki
        self._zgadnij()

    def _zgadnij(self):
        data = self.plansza.flatten().reshape(1, WEJSCIA)
        wynik = self.siec.forward(data)[0]
        idx = np.argmax(wynik)
        pewnosc = wynik[idx]
        
        znak = ZNAKI[idx]
        self.lbl_wynik.config(text=znak)
        self.lbl_detale.config(text=f"Pewność: {pewnosc:.2%}")
        self._aktualizuj_wizualizacje_live()

    def _reset(self):
        self.canvas.delete("all")
        self.plansza = np.zeros((ROZMIAR_SIATKI, ROZMIAR_SIATKI))
        self.lbl_wynik.config(text="?")
        # Reset wizualizacji
        for nid in self.hidden_ids: self.viz_canvas.itemconfig(nid, fill=COL_NEURON_OFF)
        for nid in self.output_ids: self.viz_canvas.itemconfig(nid, fill=COL_NEURON_OFF)

    def _ucz_recznie(self):
        znak = self.var_znak.get()
        idx = ZNAKI.index(znak)
        target = np.zeros((1, WYJSCIA))
        target[0][idx] = 1
        
        data = self.plansza.flatten().reshape(1, WEJSCIA)
        self.siec.backward(data, target)
        self.lbl_status.config(text=f"Nauczono ręcznie: {znak}", fg="#00ff00")
        self._aktualizuj_wizualizacje_live()

    def _start_mnist(self):
        self.lbl_status.config(text="Pobieranie MNIST...", fg="yellow")
        self.root.update()
        
        path = "mnist.npz"
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", path)
            except:
                messagebox.showerror("Błąd", "Brak internetu!")
                return

        with np.load(path, allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
        
        # Trening na próbce 1500 (dla szybkości)
        limit = 1500
        x_data = x_train[:limit].reshape(limit, 784) / 255.0
        y_labels = y_train[:limit]
        
        self.lbl_status.config(text="Trening w toku...", fg="yellow")
        
        for i in range(limit):
            target = np.zeros((1, WYJSCIA))
            target[0][y_labels[i]] = 1 # Mapuje cyfrę na odpowiednie wyjście
            
            # Podajemy dane jako macierz (1, 784)
            img = x_data[i:i+1]
            self.siec.backward(img, target)
            
            if i % 100 == 0:
                self.root.update()
        
        self.lbl_status.config(text="Trening MNIST zakończony!", fg="#00ff00")
        messagebox.showinfo("Sukces", "Sieć wytrenowana na cyfrach 0-9")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()