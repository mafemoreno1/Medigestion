import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

np.random.seed(42)
n = 12000

# Simulación de datos clínicos
edad = np.random.randint(18, 90, n)
genero = np.random.choice(['F', 'M'], n)
tipo_cita = np.random.choice(['control', 'fisioterapia', 'laboratorio'], n, p=[0.5, 0.2, 0.3])
dia_semana = np.random.choice(['lunes', 'martes', 'miércoles', 'jueves', 'viernes'], n)
recordatorio = np.random.choice([0, 1], n, p=[0.35, 0.65])

df = pd.DataFrame({
    'edad': edad,
    'genero': genero,
    'tipo_cita': tipo_cita,
    'dia_semana': dia_semana,
    'recordatorio_previo': recordatorio
})

# Calcular riesgo CONTINUO con factores de riesgo Y protección
riesgo = (
    # Factores de RIESGO
    (df['edad'] > 65).astype(float) * 0.30 +           
    (df['edad'] < 30).astype(float) * (-0.15) +        
    (df['recordatorio_previo'] == 0).astype(float) * 0.25 +  
    (df['tipo_cita'] == 'laboratorio').astype(float) * 0.20 + 
    (df['genero'] == 'M').astype(float) * 0.05 +       
    
    # Factores PROTECTORES
    (df['tipo_cita'] == 'control').astype(float) * (-0.10) + 
    (df['dia_semana'] == 'lunes').astype(float) * (-0.08) +   
    (df['dia_semana'] == 'viernes').astype(float) * 0.12      
)

# Añadir ruido y acotar entre 0 y 1
riesgo = riesgo + np.random.normal(0, 0.08, n)
riesgo = np.clip(riesgo, 0, 1)

# VARIABLE OBJETIVO: riesgo continuo
y = riesgo

# Codificación One-Hot
df_encoded = pd.get_dummies(df, columns=['genero', 'tipo_cita', 'dia_semana'], drop_first=False)

columnas = [
    'edad', 'recordatorio_previo',
    'genero_F', 'genero_M',
    'tipo_cita_control', 'tipo_cita_fisioterapia', 'tipo_cita_laboratorio',
    'dia_semana_lunes', 'dia_semana_martes', 'dia_semana_miércoles',
    'dia_semana_jueves', 'dia_semana_viernes'
]

# Asegurar que todas las columnas existan
for col in columnas:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

X = df_encoded[columnas]

# Entrenar modelo de regresión
modelo = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
modelo.fit(X, y)

# Guardar
os.makedirs('modelo', exist_ok=True)
with open('modelo/modelo_no_show.pkl', 'wb') as f:
    pickle.dump(modelo, f)

print(f" Modelo entrenado. Riesgo promedio: {y.mean():.2f}")
print(f"Ejemplo: Paciente de 28 años con SMS → riesgo esperado ~0.05–0.15")
print(f"Ejemplo: Paciente de 28 años sin SMS → riesgo esperado ~0.25–0.35 (MEDIA)")