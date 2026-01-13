# ⚔️ Medieval Neural Network - Średniowieczna Sieć Neuronowa ⚔️

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Required-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📜 Opis projektu

**Medieval Neural Network** to interaktywna aplikacja do rozpoznawania ręcznie pisanych znaków (cyfry 0-9 i litery A-Z) z wykorzystaniem prostej sieci neuronowej. Projekt został zaprojektowany w **średniowiecznym stylu gothic** z:

- 🏰 Ciemną, pergaminową paletą kolorów (brązy, złoto, czerwień heraldyczna)
- ✍️ **Staroangielską czcionką** ("Old English Text MT") dla wyświetlanych liter
- 🎨 Ozdobnymi ramkami i separatorami w stylu iluminowanych manuskryptów
- 🧠 Wizualizacją sieci neuronowej na żywo
- 📚 Kodem zoptymalizowanym dla początkujących z obszernymi komentarzami

---

## 🎯 Funkcje

### ✨ Główne funkcje:

1. **Interaktywne rysowanie**
   - Rysuj cyfry i litery myszką na canvas'ie
   - Automatyczne rozpoznawanie po puszczeniu przycisku
   - Wyświetlanie pewności rozpoznania

2. **Wizualizacja sieci na żywo**
   - Zobacz jak pracują neurony w czasie rzeczywistym
   - Podświetlenie aktywnych neuronów (gradient złoty)
   - Linie pokazujące najsilniejsze połączenia (zielone = pozytywne, czerwone = negatywne)

3. **Trening na danych EMNIST**
   - Wczytywanie danych z pliku CSV
   - Tasowanie danych dla lepszego uczenia
   - Pasek postępu i statusy treningu
   - Możliwość treningu na 20 000 przykładów

4. **Styl średniowieczny**
   - Gotycka paleta kolorów (brązy, złoto, pergamin)
   - Staroangielska czcionka dla rozpoznanych znaków
   - Ozdobne ramki i separatory

---

## 📦 Wymagania

### Wymagane biblioteki:

```bash
Python 3.7+
numpy
tkinter (wbudowany w Python)
pandas (opcjonalnie, tylko do treningu)
```

### Instalacja zależności:

```bash
# NumPy (wymagany)
pip install numpy

# Pandas (opcjonalnie, tylko dla treningu na CSV)
pip install pandas
```

---

## 🚀 Uruchomienie

### Podstawowe uruchomienie:

```bash
python neuralnetwork.py
```

### Uruchomienie z wirtualnym środowiskiem:

```bash
# Utwórz wirtualne środowisko
python -m venv venv

# Aktywuj środowisko
# Na Windows:
venv\Scripts\activate
# Na Linux/Mac:
source venv/bin/activate

# Zainstaluj zależności
pip install numpy pandas

# Uruchom aplikację
python neuralnetwork.py
```

---

## 📖 Instrukcja użycia

### 1️⃣ Rysowanie i rozpoznawanie

1. **Narysuj znak** na lewym panelu (czarny canvas):
   - Użyj lewego przycisku myszy do rysowania
   - Rysuj grubym złotym pędzlem

2. **Puść przycisk myszy**:
   - Sieć automatycznie rozpozna znak
   - Wynik pojawi się na prawym panelu (duża staroangielska czcionka)
   - Zobaczysz pewność rozpoznania w procentach

3. **Wyczyść canvas**:
   - Kliknij prawym przyciskiem myszy
   - Lub użyj przycisku "🗡️ WYCZYŚĆ"

### 2️⃣ Wizualizacja sieci

- **Środkowy panel** pokazuje jak pracuje sieć:
  - Lewa kolumna: neurony wejściowe (20 reprezentacyjnych z 784)
  - Środek: 140 neuronów ukrytych (w siatce)
  - Prawa kolumna: 36 neuronów wyjściowych (0-9, A-Z)

- **Kolory neuronów**:
  - Ciemny brąz = nieaktywny
  - Gradient złoty = aktywny
  - Jasne złoto = zwycięski neuron

- **Linie połączeń**:
  - Zielone = pozytywne wagi
  - Czerwone = negatywne wagi
  - Grubość = siła połączenia

### 3️⃣ Trening sieci

1. **Pobierz dane EMNIST**:
   - Pobierz `emnist-balanced-train.csv`
   - Link: [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

2. **Uruchom trening**:
   - Kliknij "⚡ WCZYTAJ I TRENUJ ⚡"
   - Wybierz plik CSV
   - Obserwuj pasek postępu
   - Trening zajmuje 5-15 minut (20 000 przykładów)

3. **Po treningu**:
   - Sieć jest gotowa do użycia
   - Rysuj znaki i testuj dokładność

### 4️⃣ Ustawienia

- **Limiter wag** (lewy panel):
  - Kontroluje maksymalną wartość wag sieci
  - Zakres: 0.5 - 10.0
  - Domyślnie: 3.0
  - Wpływa na wizualizację połączeń

---

## 🧠 Architektura sieci neuronowej

### Struktura:

```
Warstwa wejściowa:  784 neurony (28×28 pikseli)
        ↓
Warstwa ukryta:     140 neuronów (funkcja sigmoid)
        ↓
Warstwa wyjściowa:  36 neuronów (0-9, A-Z)
```

### Techniki użyte:

- **Funkcja aktywacji**: Sigmoid
- **Algorytm uczenia**: Backpropagation (wsteczna propagacja błędu)
- **Inicjalizacja wag**: He initialization
- **Współczynnik uczenia**: 0.1
- **Clipping wag**: Zapobiega eksplozji wartości

### Kod dla początkujących:

- ✅ Obszerne komentarze w języku polskim
- ✅ Docstringi dla każdej funkcji
- ✅ Wyjaśnienia wzorów matematycznych
- ✅ Czytelna struktura kodu
- ✅ Tylko numpy + tkinter (proste zależności)

---

## 🎨 Paleta kolorów średniowiecznych

| Element | Kolor | Hex |
|---------|-------|-----|
| Tło główne | Ciemny brąz | `#1A0F0A` |
| Panele | Drewno | `#2C1810` |
| Złoto | Akcent | `#D4AF37` |
| Złoto jasne | Highlight | `#FFD700` |
| Czerwień heraldyczna | Błędy | `#8B0000` |
| Zieleń szlachetna | Sukces | `#2E8B57` |
| Tekst pergaminowy | Główny | `#F5E6D3` |

---

## 🔧 Optymalizacje

### Dla wydajności:

1. **Tylko numpy**: Operacje wektorowe zamiast pętli
2. **Wizualizacja Top 30**: Pokazuje tylko 30 najbardziej aktywnych neuronów
3. **Clipping**: Zapobiega overflow i eksplozji wartości
4. **Threading**: Trening w osobnym wątku (GUI nie zamarza)

### Dla początkujących:

1. **Komentarze**: Każda sekcja dokładnie opisana
2. **Polskie nazwy zmiennych**: `wagi`, `biasy`, `warstwa_ukryta`
3. **Docstringi**: Dokumentacja każdej funkcji
4. **Wzory matematyczne**: Wyjaśnienia algorytmów

---

## 📊 Format danych EMNIST

Plik CSV powinien mieć format:

```
etykieta, piksel_1, piksel_2, ..., piksel_784
5, 0, 0, 15, ..., 0
10, 0, 23, 45, ..., 12
...
```

- Kolumna 0: etykieta (0-46 w pełnym EMNIST, 0-35 używane)
- Kolumny 1-784: wartości pikseli (0-255)
- Bez nagłówka

**Mapowanie etykiet**:
- 0-9: cyfry 0-9
- 10-35: litery A-Z

---

## 🐛 Rozwiązywanie problemów

### Problem: Brak czcionki "Old English Text MT"

**Rozwiązanie**:
- Windows: Czcionka jest wbudowana
- Linux: Zainstaluj czcionki MS: `sudo apt install ttf-mscorefonts-installer`
- Mac: Zainstaluj Font Book → "Old English Text MT"
- Fallback: Kod automatycznie użyje czcionki domyślnej

### Problem: "No module named 'numpy'"

**Rozwiązanie**:
```bash
pip install numpy
```

### Problem: "No module named 'pandas'"

**Rozwiązanie**:
```bash
pip install pandas
```
(Pandas jest potrzebny tylko do treningu)

### Problem: Okno GUI nie otwiera się

**Rozwiązanie**:
- Sprawdź czy tkinter jest zainstalowany:
  ```bash
  python -c "import tkinter"
  ```
- Linux: Zainstaluj `python3-tk`:
  ```bash
  sudo apt install python3-tk
  ```

---

## 📝 Struktura plików

```
NeuralNetworkPK/
├── neuralnetwork.py         # Główna aplikacja
├── README.md               # Ten plik
├── .gitignore              # Ignorowane pliki
└── emnist-balanced-train.csv  # Dane treningowe (opcjonalnie)
```

---

## 🤝 Kontrybutor

Ten projekt został stworzony jako interaktywna aplikacja edukacyjna do nauki sieci neuronowych.

**Optymalizacje**:
- ✅ Średniowieczny design z gotycką estetyką
- ✅ Staroangielska czcionka dla wyświetlanych liter
- ✅ Kod zoptymalizowany i czytelny dla początkujących
- ✅ Używa czystego Pythona + numpy
- ✅ Dokładne komentarze w języku polskim

---

## 📜 Licencja

MIT License - możesz swobodnie używać, modyfikować i dystrybuować ten kod.

---

## 🎓 Edukacyjne zasoby

### Zrozumienie sieci neuronowych:

1. **Forward Pass** (linie 143-171):
   - Jak dane przepływają przez sieć
   - Funkcja sigmoid i jej rola
   - Mnożenie macierzy (numpy.dot)

2. **Backpropagation** (linie 174-218):
   - Jak sieć się uczy z błędów
   - Gradient descent (zejście gradientowe)
   - Aktualizacja wag i biasów

3. **Wizualizacja** (linie 504-563):
   - Jak interpretować aktywacje neuronów
   - Co pokazują połączenia między neuronami
   - Różnica między wagami pozytywnymi a negatywnymi

---

## ⚔️ Medieval Easter Eggs

- 🏰 Wszystkie ikony i teksty w stylu średniowiecznym
- ⚔️ Miecz jako symbol czyszczenia
- 📜 Pergamin jako motyw przewodni
- 🧠 Mózg jako symbol inteligencji
- ⚡ Błyskawica jako symbol mocy obliczeniowej

---

**Stworzono z ⚔️ w stylu średniowiecznym dla miłośników AI i historii!**
