# Prototipo IoT Biométrico

Sistema de control de acceso que combina reconocimiento facial (con verificación de parpadeo) y lectura de huellas dactilares. El proyecto se divide en dos carpetas:

- **Servidor/**: aplicación Flask que gestiona usuarios, sincroniza el dataset, entrena las codificaciones faciales y expone la interfaz web.
- **Cliente/**: script para la Raspberry Pi que captura imágenes y huellas, y se comunica por MQTT con el servidor.

## Requisitos
- Python 3.10+ con soporte para `venv`.
- Librerías de sistema necesarias para compilar `dlib` y `opencv` (por ejemplo, encabezados de C/C++).
- Archivo de modelos `shape_predictor_68_face_landmarks.dat` ubicado en `Servidor/` (obligatorio para la detección facial).
- Broker MQTT accesible (por defecto `127.0.0.1:1883`).

## Preparación del servidor
Ejecuta todo dentro de `Servidor/` usando un entorno virtual para aislar dependencias.

```bash
cd Servidor
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_server.txt
```

### Inicializar la base de datos y carpetas
El comando CLI `init-db` crea `database.db`, genera el usuario administrador y prepara el entorno para el dataset.

```bash
export FLASK_APP=app.py
flask --app app.py init-db
```

- Usuario inicial: **admin / admin**.
- Se crea `Servidor/dataset/` si no existía.
- Se envía un comando MQTT para limpiar huellas almacenadas en el sensor.

### Generar `encodings.pickle`
Asegúrate de contar con fotografías en `Servidor/dataset/<cedula>/<archivo>.jpg` antes de entrenar (el cliente puede generar estas capturas durante el enrolamiento).

Ejecuta la tarea completa de entrenamiento desde el CLI de Flask:

```bash
flask --app app.py shell <<'PY'
from app import train_encodings_task
train_encodings_task()
PY
```

Esto procesará todo el dataset y escribirá `Servidor/encodings.pickle`. El proceso también sincroniza los flags `has_facial` en la base de datos.

### Ejecutar el servidor web
Usa el mismo entorno virtual y lanza la app directamente para habilitar el listener MQTT y la UI web:

```bash
python app.py
```

La interfaz estará disponible en `http://localhost:5000`. Ingresa con el usuario administrador para gestionar altas, enrolar biometría y lanzar reentrenamientos manuales desde el panel.

## Notas sobre artefactos generados
- El dataset de fotos y el archivo `encodings.pickle` se regeneran durante el entrenamiento; mantenlos bajo `Servidor/`.
- Directorios como `__pycache__/` se crearán automáticamente al ejecutar Python; no es necesario versionarlos.

## Ejecución del cliente (Raspberry Pi)
En la Raspberry Pi instala las dependencias equivalentes para `Cliente/client_rpi.py` (bibliotecas de cámara, `paho-mqtt`, `opencv`, `pyserial`, etc.). Configura la IP del broker en el script si es distinto al valor por defecto y ejecútalo con Python para empezar a capturar y enviar eventos biométricos.
