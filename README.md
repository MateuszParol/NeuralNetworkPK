# 🧠 Neural Network OCR v9.0

Zaawansowana sieć neuronowa do rozpoznawania liter i cyfr z piękną wizualizacją w czasie rzeczywistym i animacjami pokazującymi działanie sieci.

![Neural Network OCR](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Funkcje

- 🎨 **Interaktywny Canvas** - Rysuj litery i cyfry bezpośrednio w aplikacji
- 🧠 **Zaawansowana Architektura** - Trójwarstwowa sieć (784 → 256 → 128 → 36) z ReLU i Softmax
- ⚡ **Live Wizualizacja** - Zobacz jak działa sieć neuronowa w czasie rzeczywistym
- 🌊 **Particle Animations** - Piękne animacje pokazujące przepływ sygnałów przez sieć
- 📊 **Top 5 Predictions** - Zobacz nie tylko najlepsze dopasowanie, ale 5 najbardziej prawdopodobnych wyników
- 💾 **Save/Load Model** - Zapisuj i wczytuj wytrenowane modele
- 🚀 **EMNIST Training** - Trenuj na profesjonalnym zbiorze danych EMNIST
- 📈 **Validation Accuracy** - Monitoruj dokładność podczas treningu

## 🎯 Architektura Sieci

```
Input Layer:    784 neurons (28×28 pikseli)
                    ↓
Hidden Layer 1: 256 neurons (ReLU activation)
                    ↓
Hidden Layer 2: 128 neurons (ReLU activation)
                    ↓
Output Layer:   36 neurons (Softmax) → 0-9, A-Z
```

### Cechy techniczne:
- **Inicjalizacja wag**: He initialization (optymalna dla ReLU)
- **Funkcje aktywacji**: ReLU (warstwy ukryte), Softmax (wyjście)
- **Regularizacja**: L2 regularization (λ=0.0001)
- **Gradient Clipping**: Zapobiega eksplozji gradientów
- **Adaptive Learning Rate**: Zmniejsza się podczas treningu
- **Batch Training**: Mini-batches po 32 próbki

## 🚀 Instalacja

### Wymagania
- Python 3.8 lub nowszy
- pip

### Krok 1: Klonuj repozytorium
```bash
git clone <repository-url>
cd NeuralNetworkPK
```

### Krok 2: Zainstaluj zależności
```bash
pip install -r requirements.txt
```

### Krok 3: Uruchom aplikację
```bash
python neuralnetwork.py
```

## 📖 Jak Używać

### 1. Rysowanie i Rozpoznawanie

1. **Narysuj znak** - Użyj myszki aby narysować cyfrę (0-9) lub literę (A-Z) na czarnym canvasie
2. **Automatyczne rozpoznanie** - Sieć automatycznie rozpozna znak gdy skończysz rysować
3. **Zobacz wyniki** - Panel po prawej pokazuje:
   - Rozpoznany znak (duża litera)
   - Pewność predykcji (%)
   - Top 5 najbardziej prawdopodobnych znaków

4. **Obserwuj wizualizację** - Panel środkowy pokazuje:
   - Aktywne neurony (świecące na zielono/fioletowo)
   - Połączenia synaptyczne (zielone = pozytywne wagi, czerwone = negatywne)
   - Animowane cząsteczki pokazujące przepływ sygnału

5. **Wyczyść canvas** - Kliknij "🗑️ CLEAR" aby wyczyścić i spróbować ponownie

### 2. Trening na EMNIST

#### Pobierz dataset EMNIST:
1. Odwiedź: https://www.nist.gov/itl/products-and-services/emnist-dataset
2. Pobierz **EMNIST Balanced** w formacie CSV
3. Rozpakuj plik `emnist-balanced-train.csv`

#### Trenuj model:
1. Kliknij **"📊 LOAD & TRAIN EMNIST"**
2. Wybierz plik CSV z danymi
3. Poczekaj na zakończenie treningu (około 5 epok)
4. Model zostanie automatycznie zapisany jako `model_ocr.pkl`

**Parametry treningu:**
- Batch size: 32
- Epochs: 5
- Learning rate: 0.01 (decay 0.95 per epoch)
- Train/Val split: 90%/10%
- Optymalizator: Gradient Descent z L2 regularizacją

### 3. Zapisywanie i Wczytywanie Modeli

- **💾 SAVE** - Zapisz aktualny model do pliku `model_ocr.pkl`
- **📂 LOAD** - Wczytaj wcześniej zapisany model

Model jest automatycznie wczytywany przy starcie aplikacji (jeśli `model_ocr.pkl` istnieje).

## 🎨 Wizualizacja

### Kolory i Znaczenie

**Neurony:**
- 🟢 **Zielony/Jasny** - Wysoka aktywacja (silny sygnał)
- 🔵 **Niebieski** - Neuron wyjściowy (zwycięzca)
- 🟣 **Fioletowy** - Warstwa ukryta 2
- ⚫ **Ciemny** - Niska aktywacja

**Połączenia:**
- 🟢 **Zielona linia** - Pozytywna waga (wzmacnia sygnał)
- 🔴 **Czerwona linia** - Negatywna waga (osłabia sygnał)
- **Grubość linii** - Siła połączenia (większa waga = grubsza linia)

**Animacje:**
- ✨ **Cząsteczki** - Pokazują przepływ sygnału przez sieć
- 💫 **Pulsowanie** - Zwycięski neuron pulsuje

### Toggle Animacji
Użyj checkboxa **"✨ Particle Animation"** aby włączyć/wyłączyć animacje cząsteczek.

## 📊 Statystyki

- **Neurony**: 384 (256 + 128)
- **Połączenia**: ~233,000 (200,704 + 32,768)
- **Parametry do trenowania**: ~234,000 wag + biasy
- **Oczekiwana dokładność**: 85-92% (po treningu na EMNIST)

## 🎯 Wskazówki dla Najlepszych Wyników

1. **Rysuj w centrum** - Umieszczaj znaki w środku canvasu
2. **Odpowiedni rozmiar** - Znaki nie powinny być zbyt małe ani zbyt duże
3. **Litery drukowane** - Sieć najlepiej rozpoznaje litery drukowane (nie pisane)
4. **Trenuj na danych** - Dla najlepszej dokładności, wytrenuj model na EMNIST
5. **Czyść całkowicie** - Przed rysowaniem nowego znaku wyczyść canvas

## 🔧 Konfiguracja

Możesz dostosować parametry w pliku `neuralnetwork.py`:

```python
# Architektura sieci
UKRYTE1 = 256  # Neurony w pierwszej warstwie ukrytej
UKRYTE2 = 128  # Neurony w drugiej warstwie ukrytej

# Kolory motywu
THEME = {
    "bg": "#0a0e27",
    "accent": "#00d9ff",
    # ... więcej kolorów
}

# Trening
batch_size = 32
epochs = 5
learning_rate = 0.01
```

## 🐛 Rozwiązywanie Problemów

### Aplikacja nie uruchamia się
```bash
# Sprawdź wersję Pythona
python --version  # Powinno być 3.8+

# Zainstaluj ponownie zależności
pip install --upgrade -r requirements.txt
```

### Błąd "BRAK PANDAS"
```bash
pip install pandas
```

### Niska dokładność
- Upewnij się, że model jest wytrenowany (użyj EMNIST dataset)
- Rysuj znaki wyraźnie i czytelnie
- Spróbuj wytrenować model dłużej (zwiększ liczbę epok)

### Wizualizacja jest wolna
- Wyłącz animacje cząsteczek (checkbox "Particle Animation")
- Zmniejsz liczbę wyświetlanych neuronów w kodzie

## 📝 Format Danych EMNIST

Oczekiwany format CSV:
```
label,pixel0,pixel1,pixel2,...,pixel783
0,0,0,0,...,255
1,0,15,32,...,128
...
```

- **Kolumna 0**: Etykieta (0-35: cyfry 0-9, litery A-Z)
- **Kolumny 1-784**: Wartości pikseli (0-255)
- Obraz: 28×28 pikseli (grayscale)

## 🤝 Współpraca

Zgłaszaj błędy i propozycje ulepszeń przez Issues na GitHubie.

## 📄 Licencja

MIT License - możesz swobodnie używać, modyfikować i dystrybuować ten kod.

## 🙏 Podziękowania

- **EMNIST Dataset**: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017)
- **NumPy**: Fundament obliczeń numerycznych
- **Tkinter**: Interface graficzny

---

**Stworzone z ❤️ dla miłośników AI i Machine Learning**

🌟 Jeśli podoba Ci się ten projekt, daj mu gwiazdkę na GitHubie!
