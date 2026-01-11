# Sistema Biométrico IoT (Cliente + Servidor)

Este repositorio contiene dos módulos:

- **servidor/**: API / Web (Flask) + reconocimiento/anti-spoofing + lógica de enrolamiento.
- **cliente_rpi/**: Cliente para Raspberry Pi (cámara + huella + UI touch + servo puerta) que se comunica por MQTT.

> Recomendación: usar **un solo entorno virtual (venv)** para simplificar.

---

## Requisitos

### Python
- Python **3.10.x** (recomendado: 3.10.11)

Verificar versión:

**Windows**
```bash
py -3.10 --version
```

**Linux / Raspberry Pi**
```bash
python3.10 --version
```

### MQTT Broker
Debes tener un broker MQTT accesible (por ejemplo: Mosquitto).  
El cliente y servidor deben apuntar al mismo `MQTT_BROKER_IP` y `MQTT_PORT`.

---

## 1) Crear entorno virtual (UN SOLO venv)

Ubícate en la raíz del repo (donde están `cliente_rpi/` y `servidor/`).

### Windows (PowerShell / CMD)
```bash
py -3.10 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

### Git Bash (MINGW64)
```bash
py -3.10 -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
```

### Linux / Raspberry Pi
```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

---

## 2) Instalar dependencias desde los .txt

### Instalar servidor
Desde la raíz:
```bash
pip install -r servidor/requirements_server.txt
```

### Instalar cliente (Raspberry Pi)
Desde la raíz:
```bash
pip install -r cliente_rpi/requirements_client.txt
```

> Nota: si estás en Windows y solo estás desarrollando/validando, puedes instalar igual el `requirements_client.txt`.
> **RPi.GPIO NO se instala en Windows** (es una librería específica de Raspberry Pi/Linux).
> El código del cliente está preparado con `try/except`, así que en Windows se desactiva automáticamente GPIO/servo.

---

## 3) Configuración rápida (IP/puertos)

Revisa y ajusta (si aplica) en ambos módulos:

- `servidor/app.py`
- `cliente_rpi/client_rpi.py`

Variables típicas:
- `MQTT_BROKER_IP`
- `MQTT_PORT`
- `RPI_CLIENT_ID`

> IMPORTANTE: Cliente y servidor deben coincidir en broker/puerto y en la estructura de topics.

---

## 4) Ejecutar el servidor (Flask)

Desde la raíz:
```bash
cd servidor
python app.py
```

Si tu `app.py` usa `flask run` en vez de `python app.py`, entonces:

**Windows**
```bash
set FLASK_APP=app.py
flask run
```

**Linux**
```bash
export FLASK_APP=app.py
flask run
```

---

## 5) Ejecutar el cliente (Raspberry Pi)

Desde la raíz:
```bash
cd cliente_rpi
python client_rpi.py
```

### Notas del cliente
- Servo (si aplica):
  - pin recomendado PWM: **GPIO 18 (BCM)**
- Huella:
  - puerto típico: `/dev/ttyAMA0` (depende del setup)
- Cámara:
  - usa `CAMERA_INDEX` (0 normalmente)

---

## 6) Sobre RPi.GPIO (Windows vs Raspberry Pi)

- En **Windows** fallará la instalación de `RPi.GPIO` (esto es normal).
- En **Raspberry Pi** se instala con:
  ```bash
  pip install RPi.GPIO
  ```
  o también suele estar disponible por apt:
  ```bash
  sudo apt-get update
  sudo apt-get install -y python3-rpi.gpio
  ```

El proyecto está preparado para que si `RPi.GPIO` no está disponible:
- `SERVO_OK = False`
- se desactiva el control de servo y el programa sigue funcionando.

---

## Estructura del repositorio

```
.
├── cliente_rpi/
│   ├── client_rpi.py
│   └── requirements_client.txt
├── servidor/
│   ├── app.py
│   ├── anti_spoofing.py
│   ├── requirements_server.txt
│   └── templates/
└── .gitignore
```

---

## Troubleshooting

### 1) Error instalando RPi.GPIO en Windows
✅ Normal. No se soporta en Windows.  
Solución: ignóralo o instala/ejecuta el cliente en Raspberry Pi.

### 2) OpenCV no abre la cámara
- Revisa `CAMERA_INDEX`
- En Raspberry Pi, si tu cámara no funciona con `cv2.CAP_V4L2`, prueba inicializar sin backend explícito.

### 3) No conecta a MQTT
- Revisa `MQTT_BROKER_IP` y `MQTT_PORT`
- Verifica que el broker esté corriendo y accesible por red.
