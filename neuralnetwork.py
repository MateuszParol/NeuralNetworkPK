"""
==========================================================================
   ŚREDNIOWIECZNA SIEĆ NEURONOWA - Medieval Neural Network
==========================================================================
Projekt: Rozpoznawanie ręcznie pisanych znaków (cyfry 0-9, litery A-Z)
Styl:    Medieval/Gothic overlay z staroangielską czcionką
Autor:   Zoptymalizowany kod dla początkujących
==========================================================================
"""

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import threading

# ===== SPRAWDZENIE PANDAS (opcjonalne) =====
try:
    import pandas as pd
    PANDAS_DOSTEPNY = True
except ImportError:
    PANDAS_DOSTEPNY = False
    print("⚠️ BRAK PANDAS! Zainstaluj: pip install pandas")


# =========================================================================
#                          KONFIGURACJA STYLU
# =========================================================================

# Średniowieczna paleta kolorów (Medieval Theme)
MOTYW = {
    # Tła i panele
    "tlo_glowne": "#1A0F0A",        # Ciemny brąz (deep brown)
    "panel": "#2C1810",              # Ciemny drewniany panel (dark wood)
    "panel_jasny": "#3E2723",        # Jaśniejszy panel

    # Akcenty i ozdoby
    "zloto": "#D4AF37",              # Złoto (gold)
    "zloto_jasne": "#FFD700",        # Jasne złoto (bright gold)
    "czerwien": "#8B0000",           # Ciemna czerwień (dark red)
    "zielony": "#2E8B57",            # Szlachetna zieleń (sea green)

    # Teksty
    "tekst": "#F5E6D3",              # Pergaminowy (parchment)
    "tekst_ciemny": "#8B7355",       # Brązowy tekst (brown text)
    "tekst_jasny": "#FFF8DC",        # Kukurydziany (cornsilk)

    # Wizualizacja sieci
    "neuron_aktywny": "#FFD700",     # Złoty dla aktywnych neuronów
    "neuron_nieaktywny": "#4A2511",  # Ciemny brąz dla nieaktywnych
    "polaczenie_plus": "#2E8B57",    # Zielone dla pozytywnych wag
    "polaczenie_minus": "#DC143C",   # Karmazynowe dla negatywnych wag

    # Ramki i linie
    "ramka": "#8B7355",              # Brązowa ramka
    "separator": "#D4AF37",          # Złoty separator
}


# =========================================================================
#                      PARAMETRY SIECI NEURONOWEJ
# =========================================================================

ROZMIAR_OBRAZKA = 28                    # Obrazek 28x28 pikseli
LICZBA_NEURONOW_WEJSCIOWYCH = 784       # 28 * 28 = 784 piksele
LICZBA_NEURONOW_UKRYTYCH = 140          # Warstwa ukryta (zwiększona dla lepszego uczenia)
LICZBA_NEURONOW_WYJSCIOWYCH = 36        # 10 cyfr + 26 liter = 36 znaków

# Lista wszystkich rozpoznawanych znaków
ZNAKI = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# =========================================================================
#                         KLASA SIECI NEURONOWEJ
# =========================================================================

class SiecNeuronowa:
    """
    Prosta sieć neuronowa z 2 warstwami dla początkujących.

    Architektura:
    - Warstwa wejściowa: 784 neurony (28x28 pikseli)
    - Warstwa ukryta:    140 neuronów
    - Warstwa wyjściowa: 36 neuronów (0-9, A-Z)

    Użyte techniki:
    - Funkcja aktywacji: Sigmoid
    - Algorytm uczenia: Backpropagation (wsteczna propagacja błędu)
    - Inicjalizacja wag: He initialization (dla lepszej zbieżności)
    """

    def __init__(self):
        """
        Inicjalizacja sieci neuronowej.
        Tworzymy wagi i biasy dla obu warstw.
        """
        # Ustawiamy ziarno losowości dla powtarzalności wyników
        np.random.seed(42)

        # ----- WARSTWA 1: Wejście -> Ukryta -----
        # Wagi: macierz 784x140 (każdy neuron wejściowy łączy się z każdym ukrytym)
        # He initialization: mnożymy przez sqrt(2/n_wejsc) dla lepszej zbieżności
        self.wagi1 = np.random.randn(LICZBA_NEURONOW_WEJSCIOWYCH,
                                      LICZBA_NEURONOW_UKRYTYCH) * np.sqrt(2.0 / LICZBA_NEURONOW_WEJSCIOWYCH)

        # Biasy: dodatkowe wartości dla każdego neuronu ukrytego (zaczynamy od 0)
        self.biasy1 = np.zeros((1, LICZBA_NEURONOW_UKRYTYCH))

        # ----- WARSTWA 2: Ukryta -> Wyjście -----
        # Wagi: macierz 140x36 (każdy neuron ukryty łączy się z każdym wyjściowym)
        self.wagi2 = np.random.randn(LICZBA_NEURONOW_UKRYTYCH,
                                      LICZBA_NEURONOW_WYJSCIOWYCH) * np.sqrt(2.0 / LICZBA_NEURONOW_UKRYTYCH)

        # Biasy: dodatkowe wartości dla każdego neuronu wyjściowego
        self.biasy2 = np.zeros((1, LICZBA_NEURONOW_WYJSCIOWYCH))

        # ----- PRZECHOWYWANIE STANÓW (do wizualizacji) -----
        self.warstwa_ukryta = np.zeros((1, LICZBA_NEURONOW_UKRYTYCH))
        self.warstwa_wyjsciowa = np.zeros((1, LICZBA_NEURONOW_WYJSCIOWYCH))


    def sigmoid(self, x):
        """
        Funkcja aktywacji Sigmoid: przekształca wartość do zakresu (0, 1).

        Wzór: σ(x) = 1 / (1 + e^(-x))

        Clipping (-500, 500) zapobiega overflow przy dużych wartościach.
        """
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


    def pochodna_sigmoid(self, sigmoid_x):
        """
        Pochodna funkcji sigmoid (używana w backpropagation).

        Wzór: σ'(x) = σ(x) * (1 - σ(x))

        Parametr sigmoid_x to już obliczona wartość sigmoid(x).
        """
        return sigmoid_x * (1.0 - sigmoid_x)


    def forward(self, wejscie):
        """
        PROPAGACJA W PRZÓD (Forward Pass).

        Przekazujemy dane wejściowe przez sieć i obliczamy wyjście.

        Krok 1: Wejście -> Warstwa ukryta
            z1 = wejscie * wagi1 + biasy1
            ukryta = sigmoid(z1)

        Krok 2: Warstwa ukryta -> Wyjście
            z2 = ukryta * wagi2 + biasy2
            wyjscie = sigmoid(z2)

        Parametry:
            wejscie - macierz 1x784 (spłaszczony obrazek)

        Zwraca:
            wyjscie - macierz 1x36 (prawdopodobieństwa dla każdego znaku)
        """
        # Krok 1: Obliczamy aktywację warstwy ukrytej
        z1 = np.dot(wejscie, self.wagi1) + self.biasy1
        self.warstwa_ukryta = self.sigmoid(z1)

        # Krok 2: Obliczamy aktywację warstwy wyjściowej
        z2 = np.dot(self.warstwa_ukryta, self.wagi2) + self.biasy2
        self.warstwa_wyjsciowa = self.sigmoid(z2)

        return self.warstwa_wyjsciowa


    def trenuj(self, wejscie, oczekiwane_wyjscie, wspolczynnik_uczenia=0.1, limit_wag=5.0):
        """
        UCZENIE SIECI (Backpropagation - wsteczna propagacja błędu).

        Algorytm:
        1. Forward pass - obliczamy wyjście
        2. Obliczamy błąd (różnicę między oczekiwanym a rzeczywistym wyjściem)
        3. Propagujemy błąd wstecz przez sieć
        4. Aktualizujemy wagi i biasy

        Parametry:
            wejscie               - obrazek (macierz 1x784)
            oczekiwane_wyjscie    - poprawna odpowiedź (macierz 1x36, one-hot encoded)
            wspolczynnik_uczenia  - jak szybko uczymy (domyślnie 0.1)
            limit_wag             - maksymalna wartość wag (clipping zapobiega eksplozji)
        """
        # --- KROK 1: FORWARD PASS ---
        wyjscie = self.forward(wejscie)

        # --- KROK 2: OBLICZANIE BŁĘDU WYJŚCIA ---
        # Błąd = różnica między tym co chcemy a tym co dostaliśmy
        blad_wyjscia = oczekiwane_wyjscie - wyjscie

        # Gradient wyjścia (ile trzeba zmienić neurony wyjściowe)
        gradient_wyjscia = blad_wyjscia * self.pochodna_sigmoid(wyjscie)

        # --- KROK 3: PROPAGACJA BŁĘDU DO WARSTWY UKRYTEJ ---
        # Obliczamy ile każdy neuron ukryty przyczynił się do błędu
        blad_ukrytej = gradient_wyjscia.dot(self.wagi2.T)

        # Gradient warstwy ukrytej
        gradient_ukrytej = blad_ukrytej * self.pochodna_sigmoid(self.warstwa_ukryta)

        # --- KROK 4: AKTUALIZACJA WAG I BIASÓW ---
        # Aktualizujemy wagi między warstwą ukrytą a wyjściem
        self.wagi2 += self.warstwa_ukryta.T.dot(gradient_wyjscia) * wspolczynnik_uczenia
        self.biasy2 += np.sum(gradient_wyjscia, axis=0, keepdims=True) * wspolczynnik_uczenia

        # Aktualizujemy wagi między wejściem a warstwą ukrytą
        self.wagi1 += wejscie.T.dot(gradient_ukrytej) * wspolczynnik_uczenia
        self.biasy1 += np.sum(gradient_ukrytej, axis=0, keepdims=True) * wspolczynnik_uczenia

        # --- KROK 5: CLIPPING WAG (zapobieganie zbyt dużym wartościom) ---
        self.wagi1 = np.clip(self.wagi1, -limit_wag, limit_wag)
        self.wagi2 = np.clip(self.wagi2, -limit_wag, limit_wag)


# =========================================================================
#                     KLASA APLIKACJI GUI (TKINTER)
# =========================================================================

class AplikacjaSredniowieczna:
    """
    Główna aplikacja z interfejsem graficznym w stylu średniowiecznym.

    Trzy panele:
    - LEWY:    Rysowanie znaków ręcznie
    - ŚRODEK:  Wizualizacja sieci neuronowej na żywo
    - PRAWY:   Wyniki rozpoznawania i trening
    """

    def __init__(self, root):
        """Inicjalizacja aplikacji."""
        self.root = root
        self.root.title("⚔️ Medieval Neural Network - Rozpoznawanie Znaków ⚔️")
        self.root.geometry("1400x850")
        self.root.configure(bg=MOTYW["tlo_glowne"])

        # Tworzymy sieć neuronową
        self.siec = SiecNeuronowa()

        # Plansza do rysowania (28x28 pikseli)
        self.plansza = np.zeros((ROZMIAR_OBRAZKA, ROZMIAR_OBRAZKA))

        # Budujemy interfejs
        self.buduj_interfejs()
        self.inicjuj_wizualizacje()


    def buduj_interfejs(self):
        """
        Tworzy cały interfejs graficzny (GUI) w stylu średniowiecznym.
        """
        # Główna ramka
        main = tk.Frame(self.root, bg=MOTYW["tlo_glowne"])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # ========================= LEWY PANEL =========================
        # Panel do rysowania znaków
        panel_lewy = tk.Frame(main, bg=MOTYW["panel"], width=340, relief="ridge", bd=3,
                              highlightbackground=MOTYW["ramka"], highlightthickness=2)
        panel_lewy.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        panel_lewy.pack_propagate(False)

        # Nagłówek panelu
        self.stworz_naglowek(panel_lewy, "⚔️ RYSOWANIE ⚔️")

        # Canvas do rysowania (280x280 pikseli)
        self.canvas = tk.Canvas(panel_lewy, width=280, height=280, bg="#0A0604",
                               highlightthickness=2, highlightbackground=MOTYW["zloto"])
        self.canvas.pack(pady=10)

        # Obsługa myszy
        self.canvas.bind("<B1-Motion>", self.rysuj)              # Lewy przycisk - rysowanie
        self.canvas.bind("<ButtonRelease-1>", self.koniec_rysowania)  # Puszczenie przycisku
        self.canvas.bind("<Button-3>", lambda e: self.czysc())   # Prawy przycisk - czyszczenie

        # Przycisk czyszczenia
        btn_reset = tk.Button(panel_lewy, text="🗡️ WYCZYŚĆ (PPM)", command=self.czysc,
                             bg=MOTYW["czerwien"], fg=MOTYW["tekst_jasny"],
                             font=("Trajan Pro", 11, "bold"), relief="raised", bd=3,
                             activebackground="#B22222", cursor="hand2")
        btn_reset.pack(fill=tk.X, padx=20, pady=10)

        # Separator
        self.stworz_separator(panel_lewy)

        # Suwak limitera wag
        tk.Label(panel_lewy, text="⚙️ LIMITER WAG", fg=MOTYW["zloto"],
                bg=MOTYW["panel"], font=("Trajan Pro", 10, "bold")).pack(pady=(20,5))

        tk.Label(panel_lewy, text="(Maksymalna wartość wag sieci)",
                fg=MOTYW["tekst_ciemny"], bg=MOTYW["panel"],
                font=("Arial", 8)).pack(pady=(0,5))

        self.var_limit = tk.DoubleVar(value=3.0)
        style = ttk.Style()
        style.theme_use('default')
        style.configure("Medieval.Horizontal.TScale",
                       background=MOTYW["panel"],
                       troughcolor=MOTYW["panel_jasny"],
                       borderwidth=2,
                       relief="raised")

        suwak = ttk.Scale(panel_lewy, from_=0.5, to=10.0, variable=self.var_limit,
                         style="Medieval.Horizontal.TScale")
        suwak.pack(fill=tk.X, padx=30, pady=5)

        # Wyświetlanie wartości suwaka
        self.label_limit = tk.Label(panel_lewy, text=f"Wartość: {self.var_limit.get():.1f}",
                                   fg=MOTYW["tekst"], bg=MOTYW["panel"],
                                   font=("Arial", 9))
        self.label_limit.pack()

        # Aktualizacja labela przy zmianie suwaka
        self.var_limit.trace_add("write", lambda *args: self.label_limit.config(
            text=f"Wartość: {self.var_limit.get():.1f}"))


        # ========================= ŚRODKOWY PANEL =========================
        # Panel wizualizacji sieci neuronowej
        panel_srodek = tk.Frame(main, bg=MOTYW["tlo_glowne"])
        panel_srodek.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        # Nagłówek
        tk.Label(panel_srodek, text="🧠 ŚCIEŻKI NEURONOWE (NA ŻYWO) 🧠",
                fg=MOTYW["zloto"], bg=MOTYW["tlo_glowne"],
                font=("Trajan Pro", 12, "bold")).pack(pady=10)

        # Canvas wizualizacji
        ramka_vis = tk.Frame(panel_srodek, bg=MOTYW["panel"], relief="ridge", bd=3,
                            highlightbackground=MOTYW["ramka"], highlightthickness=2)
        ramka_vis.pack(fill=tk.BOTH, expand=True)

        self.vis_canvas = tk.Canvas(ramka_vis, bg=MOTYW["panel"], highlightthickness=0)
        self.vis_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


        # ========================= PRAWY PANEL =========================
        # Panel wyników i treningu
        panel_prawy = tk.Frame(main, bg=MOTYW["panel"], width=340, relief="ridge", bd=3,
                              highlightbackground=MOTYW["ramka"], highlightthickness=2)
        panel_prawy.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        panel_prawy.pack_propagate(False)

        # Nagłówek
        self.stworz_naglowek(panel_prawy, "📜 ROZPOZNANO 📜")

        # Wyświetlacz rozpoznanego znaku (duża staroangielska czcionka)
        self.label_wynik = tk.Label(panel_prawy, text="?",
                                   font=("Old English Text MT", 100, "bold"),
                                   fg=MOTYW["zloto_jasne"], bg=MOTYW["panel"])
        self.label_wynik.pack(pady=10)

        # Pewność rozpoznania
        self.label_pewnosc = tk.Label(panel_prawy, text="Pewność: 0%",
                                     fg=MOTYW["tekst"], bg=MOTYW["panel"],
                                     font=("Trajan Pro", 11))
        self.label_pewnosc.pack()

        # Separator
        self.stworz_separator(panel_prawy)

        # Sekcja treningu
        tk.Label(panel_prawy, text="🏰 TRENING SIECI 🏰",
                fg=MOTYW["zloto"], bg=MOTYW["panel"],
                font=("Trajan Pro", 12, "bold")).pack(pady=(20,5))

        tk.Label(panel_prawy, text="Wymagany plik CSV:\nemnist-balanced-train.csv",
                fg=MOTYW["tekst_ciemny"], bg=MOTYW["panel"],
                font=("Arial", 9), justify=tk.CENTER).pack(pady=5)

        # Przycisk treningu
        self.btn_trenuj = tk.Button(panel_prawy, text="⚡ WCZYTAJ I TRENUJ ⚡",
                                    command=self.start_treningu,
                                    bg=MOTYW["zielony"], fg=MOTYW["tekst_jasny"],
                                    font=("Trajan Pro", 11, "bold"), relief="raised", bd=3,
                                    activebackground="#3CB371", cursor="hand2", pady=10)
        self.btn_trenuj.pack(fill=tk.X, padx=20, pady=10)

        # Pasek postępu
        style.configure("Medieval.Horizontal.TProgressbar",
                       troughcolor=MOTYW["panel_jasny"],
                       background=MOTYW["zloto"],
                       darkcolor=MOTYW["zloto"],
                       lightcolor=MOTYW["zloto_jasne"],
                       borderwidth=2,
                       relief="raised")

        self.progress = ttk.Progressbar(panel_prawy, orient="horizontal", length=280,
                                       mode="determinate", style="Medieval.Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, padx=20, pady=5)

        # Status treningu
        self.label_status = tk.Label(panel_prawy,
                                    text="⌛ Oczekiwanie na dane...",
                                    fg=MOTYW["tekst"], bg=MOTYW["panel"],
                                    wraplength=300, font=("Arial", 10), justify=tk.LEFT)
        self.label_status.pack(pady=10, padx=10)

        # Dodatkowe informacje
        self.stworz_separator(panel_prawy)

        info_text = ("📖 INSTRUKCJA:\n\n"
                    "1. Narysuj znak na planszy\n"
                    "2. Sieć automatycznie rozpozna\n"
                    "3. Wczytaj CSV i trenuj sieć\n"
                    "4. Obserwuj wizualizację!")

        tk.Label(panel_prawy, text=info_text, fg=MOTYW["tekst_ciemny"],
                bg=MOTYW["panel"], font=("Arial", 9), justify=tk.LEFT).pack(pady=10, padx=15)


    def stworz_naglowek(self, parent, tekst):
        """Tworzy ozdobny nagłówek w stylu średniowiecznym."""
        ramka = tk.Frame(parent, bg=MOTYW["zloto"], height=3)
        ramka.pack(fill=tk.X, pady=(15,5))

        label = tk.Label(parent, text=tekst, fg=MOTYW["zloto_jasne"],
                        bg=MOTYW["panel"], font=("Trajan Pro", 13, "bold"))
        label.pack(pady=5)

        ramka2 = tk.Frame(parent, bg=MOTYW["zloto"], height=3)
        ramka2.pack(fill=tk.X, pady=(5,15))


    def stworz_separator(self, parent):
        """Tworzy ozdobny separator."""
        sep = tk.Frame(parent, height=2, bg=MOTYW["separator"], relief="raised", bd=1)
        sep.pack(fill=tk.X, pady=20, padx=30)


    def inicjuj_wizualizacje(self):
        """
        Inicjalizuje pozycje neuronów na canvas'ie wizualizacji.

        Tworzymy trzy kolumny neuronów:
        - Lewa:   20 neuronów wejściowych (reprezentacja 784)
        - Środek: 140 neuronów ukrytych (w siatce)
        - Prawa:  36 neuronów wyjściowych (z etykietami)
        """
        # Czekamy aż canvas się wyrenderuje
        self.root.update()
        szerokosc = self.vis_canvas.winfo_width()
        wysokosc = self.vis_canvas.winfo_height()

        # ----- NEURONY WEJŚCIOWE (lewa kolumna, 20 reprezentacyjnych) -----
        self.koordy_wejsciowe = []
        odstep_y = wysokosc / 22  # Równomiernie rozmieszczone
        for i in range(20):
            x = 40
            y = odstep_y * (i + 1) + 20
            self.koordy_wejsciowe.append((x, y))
            # Rysujemy małe kółka
            self.vis_canvas.create_oval(x-3, y-3, x+3, y+3,
                                       fill=MOTYW["neuron_nieaktywny"],
                                       outline=MOTYW["ramka"], width=1)

        # ----- NEURONY UKRYTE (środek, siatka 14x10) -----
        self.koordy_ukryte = []
        siatka_x = 14  # Kolumny
        siatka_y = 10  # Wiersze
        start_x = szerokosc // 2 - 60
        start_y = 40
        odstep_x = 12
        odstep_y_ukryte = (wysokosc - 80) / siatka_y

        for i in range(LICZBA_NEURONOW_UKRYTYCH):
            kolumna = i // siatka_y
            wiersz = i % siatka_y
            x = start_x + kolumna * odstep_x
            y = start_y + wiersz * odstep_y_ukryte
            self.koordy_ukryte.append((x, y))
            # Rysujemy neurony ukryte
            self.vis_canvas.create_oval(x-4, y-4, x+4, y+4,
                                       fill=MOTYW["neuron_nieaktywny"],
                                       outline=MOTYW["ramka"], width=1,
                                       tags=f"ukryty_{i}")

        # ----- NEURONY WYJŚCIOWE (prawa kolumna, 36 z etykietami) -----
        self.koordy_wyjsciowe = []
        odstep_y_wyjscie = wysokosc / 38
        for i in range(LICZBA_NEURONOW_WYJSCIOWYCH):
            x = szerokosc - 70
            y = odstep_y_wyjscie * (i + 1)
            self.koordy_wyjsciowe.append((x, y))

            # Rysujemy neuron
            self.vis_canvas.create_oval(x-6, y-6, x+6, y+6,
                                       fill=MOTYW["neuron_nieaktywny"],
                                       outline=MOTYW["ramka"], width=1,
                                       tags=f"wyjscie_{i}")

            # Etykieta znaku (staroangielska czcionka dla większych znaków)
            self.vis_canvas.create_text(x+25, y, text=ZNAKI[i],
                                       fill=MOTYW["tekst_ciemny"],
                                       font=("Old English Text MT", 12, "bold"),
                                       tags=f"etykieta_{i}")


    def aktualizuj_wizualizacje(self):
        """
        Aktualizuje wizualizację sieci na żywo.

        - Podświetla aktywne neurony ukryte (top 30)
        - Rysuje linie połączeń do zwycięskiego neuronu wyjściowego
        - Koloruje neurony i linie według aktywacji i wag
        """
        # Usuwamy stare linie
        self.vis_canvas.delete("linia")

        # Pobieramy limit wag z suwaka
        limit = self.var_limit.get()

        # Znajdujemy top 30 najbardziej aktywnych neuronów ukrytych
        aktywnosci_ukrytych = self.siec.warstwa_ukryta[0]
        top_indeksy = np.argsort(aktywnosci_ukrytych)[-30:]

        # Znajdujemy zwycięski neuron wyjściowy (największa aktywacja)
        indeks_zwycieskiego = np.argmax(self.siec.warstwa_wyjsciowa[0])

        # ----- RYSUJEMY POŁĄCZENIA DLA AKTYWNYCH NEURONÓW -----
        for idx_ukryty in top_indeksy:
            aktywnosc = aktywnosci_ukrytych[idx_ukryty]

            # Kolor neuronu ukrytego (gradient od ciemnego do złotego)
            intensywnosc = int(aktywnosc * 255)
            kolor_neuronu = f"#{intensywnosc//2:02x}{intensywnosc:02x}{0:02x}"
            self.vis_canvas.itemconfig(f"ukryty_{idx_ukryty}", fill=kolor_neuronu)

            # Rysujemy linię do zwycięskiego neuronu wyjściowego
            waga = self.siec.wagi2[idx_ukryty][indeks_zwycieskiego]
            grubosc = abs(waga) * 2.5 / limit

            # Rysujemy tylko jeśli linia jest wystarczająco widoczna
            if grubosc > 0.3:
                kolor_linii = MOTYW["polaczenie_plus"] if waga > 0 else MOTYW["polaczenie_minus"]
                x1, y1 = self.koordy_ukryte[idx_ukryty]
                x2, y2 = self.koordy_wyjsciowe[indeks_zwycieskiego]

                self.vis_canvas.create_line(x1, y1, x2, y2,
                                           fill=kolor_linii, width=grubosc,
                                           tags="linia")

        # ----- PODŚWIETLAMY ZWYCIĘSKI NEURON WYJŚCIOWY -----
        for i in range(LICZBA_NEURONOW_WYJSCIOWYCH):
            aktywnosc_wyjscia = self.siec.warstwa_wyjsciowa[0][i]

            if i == indeks_zwycieskiego:
                # Zwycięski neuron - złoty
                self.vis_canvas.itemconfig(f"wyjscie_{i}", fill=MOTYW["zloto_jasne"])
                self.vis_canvas.itemconfig(f"etykieta_{i}",
                                          fill=MOTYW["tekst_jasny"],
                                          font=("Old English Text MT", 14, "bold"))
            else:
                # Pozostałe neurony - ciemne
                self.vis_canvas.itemconfig(f"wyjscie_{i}", fill=MOTYW["neuron_nieaktywny"])
                self.vis_canvas.itemconfig(f"etykieta_{i}",
                                          fill=MOTYW["tekst_ciemny"],
                                          font=("Old English Text MT", 11))


    def rysuj(self, event):
        """
        Obsługa rysowania myszką na canvas'ie.

        Rysujemy grube linie i jednocześnie zapisujemy do macierzy 28x28.
        """
        x, y = event.x, event.y

        # Rysujemy na canvas'ie (grube kreski)
        r = 8  # Promień pędzla
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=MOTYW["zloto"], outline="")

        # Przekształcamy współrzędne canvas'a (280x280) na macierz (28x28)
        skala = 280 / ROZMIAR_OBRAZKA
        px = int(x / skala)
        py = int(y / skala)

        # Zapisujemy do macierzy z małym rozmyciem (3x3)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < ROZMIAR_OBRAZKA and 0 <= ny < ROZMIAR_OBRAZKA:
                    self.plansza[ny][nx] = min(1.0, self.plansza[ny][nx] + 0.7)


    def koniec_rysowania(self, event):
        """
        Wywoływane po puszczeniu przycisku myszy.
        Uruchamia rozpoznawanie narysowanego znaku.
        """
        self.rozpoznaj()


    def czysc(self):
        """Czyści canvas i macierz planszy."""
        self.canvas.delete("all")
        self.plansza = np.zeros((ROZMIAR_OBRAZKA, ROZMIAR_OBRAZKA))
        self.label_wynik.config(text="?")
        self.label_pewnosc.config(text="Pewność: 0%")


    def rozpoznaj(self):
        """
        Rozpoznaje narysowany znak używając sieci neuronowej.

        Kroki:
        1. Spłaszcza macierz 28x28 do wektora 1x784
        2. Przepuszcza przez sieć (forward pass)
        3. Wybiera neuron z największą aktywacją
        4. Wyświetla wynik i pewność
        5. Aktualizuje wizualizację
        """
        # Spłaszczamy macierz do wektora
        wejscie = self.plansza.flatten().reshape(1, -1)

        # Forward pass przez sieć
        wyjscie = self.siec.forward(wejscie)

        # Znajdujemy zwycięski neuron (indeks z największą wartością)
        indeks_zwycieskiego = np.argmax(wyjscie)
        pewnosc = wyjscie[0][indeks_zwycieskiego] * 100

        # Pobieramy znak
        rozpoznany_znak = ZNAKI[indeks_zwycieskiego]

        # Aktualizujemy wyświetlacz
        self.label_wynik.config(text=rozpoznany_znak)
        self.label_pewnosc.config(text=f"Pewność: {pewnosc:.1f}%")

        # Aktualizujemy wizualizację sieci
        self.aktualizuj_wizualizacje()


    # =========================================================================
    #                           TRENING SIECI (EMNIST)
    # =========================================================================

    def start_treningu(self):
        """
        Uruchamia okno wyboru pliku CSV i rozpoczyna trening w osobnym wątku.
        """
        # Sprawdzamy czy pandas jest dostępny
        if not PANDAS_DOSTEPNY:
            messagebox.showerror("Błąd",
                               "Biblioteka pandas nie jest zainstalowana!\n\n"
                               "Zainstaluj poleceniem:\npip install pandas")
            return

        # Otwieramy okno wyboru pliku
        plik_csv = filedialog.askopenfilename(
            title="Wybierz plik CSV z danymi EMNIST",
            filetypes=[("Pliki CSV", "*.csv"), ("Wszystkie pliki", "*.*")]
        )

        if not plik_csv:
            return  # Użytkownik anulował

        # Uruchamiamy trening w osobnym wątku (żeby GUI nie zamarzło)
        watek = threading.Thread(target=self.proces_treningu, args=(plik_csv,))
        watek.daemon = True
        watek.start()


    def proces_treningu(self, sciezka_pliku):
        """
        Główna funkcja treningu sieci na danych EMNIST.

        EMNIST to rozszerzony MNIST zawierający:
        - Cyfry 0-9
        - Litery A-Z (wielkie i małe)

        Kroki:
        1. Wczytanie danych z CSV
        2. Filtrowanie (tylko znaki 0-35)
        3. Tasowanie danych (shuffle)
        4. Trening na każdym przykładzie
        5. Aktualizacja postępu

        Parametry:
            sciezka_pliku - ścieżka do pliku CSV z danymi
        """
        try:
            # ----- KROK 1: WCZYTANIE CSV -----
            self.label_status.config(text="📚 Wczytywanie danych z CSV...", fg=MOTYW["zloto_jasne"])
            self.btn_trenuj.config(state="disabled")

            # Wczytujemy CSV (bez nagłówka)
            # Kolumna 0: etykieta (0-46)
            # Kolumny 1-785: piksele obrazka (28x28 = 784)
            dane = pd.read_csv(sciezka_pliku, header=None)

            etykiety = dane.iloc[:, 0].values  # Pierwsza kolumna
            obrazki = dane.iloc[:, 1:].values  # Pozostałe kolumny

            self.label_status.config(text=f"✅ Wczytano {len(etykiety)} przykładów")

            # ----- KROK 2: FILTROWANIE -----
            # EMNIST ma 47 klas (0-46), ale my używamy tylko 36 (0-35):
            # 0-9: cyfry
            # 10-35: litery A-Z
            self.label_status.config(text="🔍 Filtrowanie danych...", fg=MOTYW["zloto_jasne"])

            maska = etykiety < LICZBA_NEURONOW_WYJSCIOWYCH
            etykiety = etykiety[maska]
            obrazki = obrazki[maska]

            self.label_status.config(text=f"✅ Przefiltrowano: {len(etykiety)} przykładów (0-35)")

            # ----- KROK 3: TASOWANIE (SHUFFLE) -----
            # Mieszamy dane, żeby sieć uczyła się równomiernie
            # (bez tasowania mogłaby się "przeprogramować" ucząc się najpierw cyfr, potem liter)
            self.label_status.config(text="🔀 Tasowanie danych...", fg=MOTYW["zloto_jasne"])

            losowa_kolejnosc = np.random.permutation(len(etykiety))
            etykiety = etykiety[losowa_kolejnosc]
            obrazki = obrazki[losowa_kolejnosc]

            # ----- KROK 4: PRZYGOTOWANIE DANYCH DO TRENINGU -----
            # Normalizujemy piksele do zakresu 0-1
            obrazki = obrazki / 255.0

            # Bierzemy podzbiór danych (np. 20000 przykładów)
            # (pełny EMNIST ma ~100k+ przykładów, co trwa bardzo długo)
            liczba_przykladow = min(20000, len(etykiety))
            etykiety = etykiety[:liczba_przykladow]
            obrazki = obrazki[:liczba_przykladow]

            self.label_status.config(text=f"🎯 Rozpoczynam trening na {liczba_przykladow} przykładach...")
            self.progress["maximum"] = liczba_przykladow
            self.progress["value"] = 0

            # ----- KROK 5: TRENING -----
            wspolczynnik_uczenia = 0.1
            limit_wag = self.var_limit.get()

            for i in range(liczba_przykladow):
                # Pobieramy pojedynczy przykład
                etykieta = etykiety[i]
                obrazek = obrazki[i].reshape(1, -1)  # Spłaszczamy do 1x784

                # Tworzymy oczekiwane wyjście (one-hot encoding)
                # Przykład: dla etykiety 5 -> [0,0,0,0,0,1,0,0,...,0]
                oczekiwane = np.zeros((1, LICZBA_NEURONOW_WYJSCIOWYCH))
                oczekiwane[0][etykieta] = 1.0

                # Trenujemy sieć na tym przykładzie
                self.siec.trenuj(obrazek, oczekiwane,
                                wspolczynnik_uczenia=wspolczynnik_uczenia,
                                limit_wag=limit_wag)

                # Aktualizujemy postęp co 100 przykładów
                if i % 100 == 0:
                    self.progress["value"] = i
                    procent = (i / liczba_przykladow) * 100
                    self.label_status.config(
                        text=f"⚔️ Trening w toku: {i}/{liczba_przykladow} ({procent:.1f}%)\n"
                             f"Znak: {ZNAKI[etykieta]}",
                        fg=MOTYW["zloto_jasne"]
                    )
                    self.root.update()

            # ----- KROK 6: ZAKOŃCZENIE -----
            self.progress["value"] = liczba_przykladow
            self.label_status.config(
                text=f"🎉 TRENING ZAKOŃCZONY!\n\n"
                     f"Przeszkolono {liczba_przykladow} przykładów.\n"
                     f"Sieć jest gotowa do rozpoznawania!",
                fg=MOTYW["zielony"]
            )
            self.btn_trenuj.config(state="normal")

            messagebox.showinfo("Sukces!",
                              f"Trening zakończony pomyślnie!\n\n"
                              f"Przeszkolono {liczba_przykladow} przykładów.\n"
                              f"Możesz teraz rysować znaki i testować sieć.")

        except FileNotFoundError:
            self.label_status.config(text="❌ Błąd: Nie znaleziono pliku!", fg=MOTYW["czerwien"])
            self.btn_trenuj.config(state="normal")
            messagebox.showerror("Błąd", f"Nie można znaleźć pliku:\n{sciezka_pliku}")

        except Exception as e:
            self.label_status.config(text=f"❌ Błąd: {str(e)}", fg=MOTYW["czerwien"])
            self.btn_trenuj.config(state="normal")
            messagebox.showerror("Błąd podczas treningu",
                               f"Wystąpił błąd:\n\n{str(e)}\n\n"
                               f"Upewnij się, że plik CSV ma poprawny format EMNIST.")


# =========================================================================
#                          URUCHOMIENIE APLIKACJI
# =========================================================================

if __name__ == "__main__":
    # Tworzymy główne okno Tkinter
    root = tk.Tk()

    # Tworzymy aplikację
    aplikacja = AplikacjaSredniowieczna(root)

    # Uruchamiamy pętlę zdarzeń (event loop)
    root.mainloop()
