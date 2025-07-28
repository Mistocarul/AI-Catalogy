# 🐱 Proiect Machine Learning – Clasificare Rase de Pisici

Acest proiect își propune să clasifice rasa pisicilor pe baza unui set de atribute comportamentale și de mediu, folosind metode de machine learning, rețele neuronale și modele NLP.

---

## 📌 Funcționalități principale

- ✅ Preprocesarea datelor (curățare, normalizare, echilibrare)
- ✅ Antrenarea modelelor de rețele neuronale (manual & cu Keras)
- ✅ Extragerea atributelor din descrieri text
- ✅ Clasificarea rasei pisicii pe baza atributelor
- ✅ Generare de descrieri ale raselor folosind GPT-2
- ✅ Analiză de clustere și vizualizări interactive

---

## 🗂 Structura Proiectului

```bash
Proiect/
├── antrenare_retea_automat.py           # Model Keras
├── antrenare_retea_manual.py            # Rețea neuronală scrisă de la zero
├── proiect_final.py                     # Script principal cu toate funcționalitățile
├── generate_new_instances.py            # Echilibrare set de date (SMOTE)
├── generare.py                          # Generare text cu GPT-2
├── eliminare_duplicate.py               # Eliminare duplicate
├── identificare_zgomote.py              # Detectare și eliminare outliers
├── matrice_corelatie.py                 # Analiză corelații
├── schimba_in_valori_numerice.py       # Conversie date categorice
├── Dataset-Pisici.xlsx                  # Setul de date original
├── Dataset-Pisici-Modificat.xlsx        # Preprocesat
├── Dataset-Pisici-Echilibrat.xlsx       # Echilibrat
└── Dataset-Pisici-Curat.xlsx            # Curățat
