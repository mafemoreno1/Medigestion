from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Cargar modelo
modelo_path = 'modelo/modelo_no_show.pkl'
if not os.path.exists(modelo_path):
    raise FileNotFoundError("Ejecute 'entrenamiento.py' primero para generar el modelo.")

with open(modelo_path, 'rb') as f:
    modelo = pickle.load(f)

# Columnas (sin distancia_hospital, coherente con tu documento)
columnas = [
    'edad', 'recordatorio_previo',
    'genero_F', 'genero_M',
    'tipo_cita_control', 'tipo_cita_fisioterapia', 'tipo_cita_laboratorio',
    'dia_semana_lunes', 'dia_semana_martes', 'dia_semana_miércoles',
    'dia_semana_jueves', 'dia_semana_viernes'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        nombre = request.form['nombre']
        edad = int(request.form['edad'])
        genero = request.form['genero']
        tipo_cita = request.form['tipo_cita']
        dia_semana = request.form['dia_semana']
        recordatorio = 1 if 'recibio_sms' in request.form else 0

        # Crear entrada
        datos = {col: 0 for col in columnas}
        datos['edad'] = edad
        datos['recordatorio_previo'] = recordatorio
        datos[f'genero_{genero}'] = 1
        datos[f'tipo_cita_{tipo_cita}'] = 1
        datos[f'dia_semana_{dia_semana}'] = 1

        entrada = pd.DataFrame([list(datos.values())], columns=columnas)
        # Usamos PREDICT (no predict_proba) porque el modelo es de REGRESIÓN
        prob_no_show = modelo.predict(entrada)[0]
        # Asegurar que esté entre 0 y 1
        prob_no_show = max(0.0, min(1.0, prob_no_show))

        # Clasificación en tres niveles 
        if prob_no_show >= 0.60:
            resultado = f"Paciente {nombre}: Alta probabilidad de inasistencia ({prob_no_show:.2f}) — enviar doble recordatorio y se generará un enlace único para reprogramar su cita en las próximas 2 horas."
        elif prob_no_show >= 0.30:
            resultado = f"Paciente {nombre}: Probabilidad media de inasistencia ({prob_no_show:.2f}) — confirmar cita telefónicamente."
        else:
            resultado = f"Paciente {nombre}: Baja probabilidad de inasistencia ({prob_no_show:.2f}) — mantener recordatorio estándar."

        return render_template('index.html', resultado=resultado)

    except Exception as e:
        return render_template('index.html', resultado=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)