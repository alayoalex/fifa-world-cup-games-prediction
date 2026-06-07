# Tecnologías y Técnicas del Proyecto — Guía para Dev Senior

> Pensado para alguien con experiencia en software pero nuevo en ML/Data Science.
> La analogía es siempre: *"esto es como X en backend"*.

---

## 1. El problema que resuelve

Dado un partido de fútbol que **aún no se jugó**, estimar:
- ¿Quién gana? (3 opciones: local / empate / visitante)
- ¿Cuántos goles mete cada equipo?

No hay respuesta exacta — el output es una **distribución de probabilidades**.
Ejemplo: Argentina vs México → {H: 58%, D: 25%, A: 17%}

---

## 2. Stack tecnológico

### `uv` — package manager
La alternativa moderna a `pip + virtualenv`. Más rápido, resuelve dependencias mejor, genera lockfile.
Analogía: es como `pnpm` vs `npm` para Node.

### `pandas` — manipulación de datos tabulares
La tabla de una DB, pero en memoria. Un `DataFrame` es una tabla; una `Series` es una columna.
Analogía: es como trabajar con arrays de objetos en JS, pero con SQL-like operations nativas.

```python
df[df["played"] == True]          # WHERE played = true
df.groupby("team")["goals"].mean()  # GROUP BY team AVG(goals)
df.merge(other, on="team")         # JOIN
```

Formato de almacenamiento: **Parquet** (columnar, comprimido). Equivalente a tener tu tabla en un archivo binario eficiente en lugar de CSV.

### `scikit-learn` — librería de ML
La "navaja suiza" de ML en Python. Provee:
- Modelos (regresión logística, Poisson, etc.)
- Pipelines de transformación
- Métricas de evaluación

Su patrón central es: `model.fit(X_train, y_train)` → `model.predict(X_test)`
Analogía: es como una ORM donde en vez de `.save()` haces `.fit()` y en vez de `.find()` haces `.predict()`.

### `MLflow` — experiment tracking
Registra cada "corrida" de entrenamiento: qué parámetros usaste, qué métricas obtuvo, qué modelo se guardó.
Analogía: es como tener logs estructurados de tus deploys + versionado de modelos. Como combinar Datadog con un registry de Docker para modelos.

### `Streamlit` — UI web en Python puro
Convierte un script Python en una web app. No HTML, no JS, no REST API. Solo Python.
Analogía: es como Retool pero para scripts de data science. Se re-ejecuta de arriba a abajo cada vez que el usuario interactúa.

### `Plotly` — gráficos interactivos
Genera charts interactivos (zoom, hover, etc.) que se muestran en el browser.

### `scipy.stats.poisson` — distribución de Poisson
Una distribución de probabilidad que modela "cuántas veces ocurre un evento en un intervalo".
En fútbol: si un equipo mete en promedio 1.8 goles, ¿cuál es la probabilidad de que meta exactamente 0, 1, 2, 3...?

---

## 3. Conceptos de ML usados

### Feature Engineering — construir las variables de entrada

El modelo no sabe "fútbol". Solo recibe números. La clave es convertir el historial de partidos en números útiles.

Las **features** (variables) del proyecto:

| Feature | Qué mide | Cómo se calcula |
|---------|----------|-----------------|
| `elo_diff` | Diferencia de fuerza entre equipos | Sistema Elo (ver abajo) |
| `form_pts_home` | Racha reciente del local | Promedio ponderado de los últimos 10 partidos |
| `h2h_home_winrate` | Historial directo entre estos dos equipos | % de victorias del local en enfrentamientos previos |
| `fifa_rank_diff` | Diferencia de ranking FIFA | Rank visitante - Rank local (positivo = local mejor) |
| `rest_days_diff` | Diferencia de descanso | Días desde último partido de cada equipo |
| `mv_log_ratio` | Diferencia de valor de mercado | log(valor_local / valor_visitante) |
| `neutral` | ¿Terreno neutro? | Boolean — en el Mundial todos los partidos son neutrales |

**Leakage** es el error de usar datos del futuro para entrenar. Ejemplo: si calculas la forma de un equipo con partidos posteriores al que querés predecir, tu modelo "hace trampa". El proyecto lo evita con un **pase cronológico único** — procesa los partidos en orden y actualiza el estado de cada equipo solo después de cada partido.

### Sistema Elo — rating dinámico de equipos

Originalmente diseñado para ajedrez. La idea:
- Cada equipo tiene un rating (empieza en 1500)
- Ganar contra un equipo fuerte sube más tu rating que ganar contra uno débil
- La fórmula: `nuevo_rating = rating + K * (resultado_real - resultado_esperado)`

`K` varía según la importancia del partido (partidos del Mundial K=60, amistosos K=20).

El `resultado_esperado` es: `E = 1 / (1 + 10^(-Δ/400))` — el mismo que usa el ranking de ajedrez de FIDE.

Analogía: es como el MMR de League of Legends o el ELO de ajedrez online.

### Regresión Logística Multinomial — el modelo de resultado

**Regresión logística** es un clasificador binario clásico (¿sí o no?). La versión **multinomial** maneja 3+ clases (H / D / A).

¿Qué hace internamente?
1. Toma los números de las features
2. Calcula una combinación lineal: `z = w1*elo_diff + w2*form + ...`
3. Aplica `softmax` para convertir esos números en probabilidades que sumen 1

Los `w` (pesos) se aprenden durante el entrenamiento buscando minimizar el **log-loss** (qué tan "sorprendido" estaría el modelo con los resultados reales).

Analogía: es como un score de crédito bancario, pero en vez de "aprobado/rechazado" da 3 probabilidades.

### Modelo Poisson — predicción de goles

**Distribución de Poisson**: probabilidad de que un evento ocurra `k` veces si la tasa promedio es `λ`.

En fútbol:
1. Entrenas **dos regresiones Poisson** (una para goles locales, otra para visitantes)
2. Cada una predice un `λ` (tasa esperada de goles): `λ_home = 1.8`, `λ_away = 0.9`
3. Construís una **matriz de probabilidades**: `P(home=i, away=j)` para todos los marcadores 0-0 a 6-6
4. La diagonal de esa matriz = empates, el triángulo inferior = victorias locales, etc.

Ejemplo mental: si `λ_home=1.8` y `λ_away=0.9`:
- P(2-0) = Poisson(2; 1.8) × Poisson(0; 0.9) ≈ 27% × 41% ≈ 11%
- Sumás todos los 2-0, 3-0, etc. → P(victoria local)

### Ensemble — combinar modelos

En lugar de elegir el "mejor" modelo, combinás sus predicciones.

```
p_final = 0.55 * p_logistic + 0.45 * p_poisson
```

¿Por qué funciona? Cada modelo captura señales distintas:
- **Logística**: mejor calibrada para la decisión H/D/A directa
- **Poisson**: capta la distribución de goles, útil para marcadores extremos

Si los errores de cada modelo son **no correlacionados**, el ensemble es más robusto.
Analogía: es como hacer code review con dos personas — si cometen errores distintos, combinando sus opiniones se obtiene mejor resultado.

### Validación Temporal (Walk-Forward)

El error clásico en ML con datos de series de tiempo: usar K-Fold aleatorio.

Problema: si entrenas con partidos del 2020 y testeas con partidos del 2019, el modelo "sabe el futuro". Las métricas parecen buenas pero en producción fallan.

**Solución**: para cada año de validación Y:
- Train: todos los partidos anteriores a Y
- Val: partidos del año Y

Se llama "expanding window" porque el conjunto de entrenamiento crece con cada fold.

```
fold 2018:  train=[1872..2017]  val=[2018]
fold 2019:  train=[1872..2018]  val=[2019]
fold 2021:  train=[1872..2020]  val=[2021]
...
```

Analogía: es como hacer backtesting en trading financiero — nunca entrenas con datos del "futuro" relativo.

### Pipeline de scikit-learn

En lugar de aplicar transformaciones a mano, encadenás pasos:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # rellenar NaN con mediana
    ("scaler", StandardScaler()),                    # normalizar a media=0, std=1
    ("clf", LogisticRegression()),                   # el modelo
])
```

La ventaja clave: **fit solo en train, transform en val**. El imputer aprende la mediana del train, no del val. Si no usaras Pipeline podrías filtrar estadísticas del val al train (leakage).

Analogía: es como un middleware chain en Express — cada paso transforma el dato antes de pasarlo al siguiente.

---

## 4. Arquitectura del sistema

```
Fuentes externas               ETL                  Features            Modelos             UI
─────────────────        ─────────────────        ──────────        ──────────────       ──────────
martj42 (GitHub)  ──▶   sources/              
FIFA Rankings     ──▶   make_dataset.py  ──▶  build_features.py  ──▶  logistic.py  ──▶
Transfermarkt     ──▶   build_match_table.py       (Parquet)           poisson.py        app.py
WC 2026 fixtures  ──▶   team_names.py                               ensemble.py      (Streamlit)
                                                                    predict_all.py
```

**Flujo de datos**:
1. `make_dataset.py` descarga CSVs de GitHub → `data/raw/`
2. `build_match_table.py` estandariza nombres y formatos → `data/interim/matches.parquet`
3. `build_features.py` hace el pase cronológico → `data/processed/matches_features.parquet`
4. `predict_all.py` entrena los 3 modelos y genera predicciones → `data/processed/wc2026_predictions_full.csv`
5. `app.py` (Streamlit) muestra los CSVs y permite ejecutar los pasos anteriores desde la UI

---

## 5. Métricas de evaluación

### Log-loss (cross-entropy)
Penaliza predicciones confiadas que resultan incorrectas más que predicciones inciertas.
Si decís "H con 95%" y gana el visitante, el log-loss es mucho peor que si dijiste "H con 40%".
Valor más bajo = mejor. Random guessing ≈ 1.1 para 3 clases.

### Accuracy
% de predicciones correctas (tomando la clase con mayor probabilidad).
Contexto: ~61% es bueno para fútbol — el deporte es inherentemente impredecible.

### MAE (Mean Absolute Error)
Para goles: promedio de `|goles_predichos - goles_reales|`.
Un MAE de 1.05 goles locales significa que en promedio te equivocás por 1 gol.

---

## 6. Limitaciones conocidas del modelo

| Limitación | Impacto | Posible mejora |
|------------|---------|----------------|
| Poisson asume independencia home/away | Subestima empates | Dixon-Coles model |
| FIFA ranking: solo 2 actualizaciones/año | Feature con poca granularidad | Elo ya lo suple bien |
| Market value: snapshot actual, no histórico | Partidos viejos tienen datos "del futuro" | Solo usarlo para fixtures recientes |
| No hay datos de jugadores individuales | No captura lesiones, convocatorias | Integrar con sofascore/statsbomb |
| Hiperparámetros fijos (C=1.0, α=0.1) | Modelo posiblemente no óptimo | Optuna para búsqueda automática |
