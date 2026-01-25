import sys

# Deshabilitar buffering de stdout para ver logs en tiempo real
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
else:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

print("Iniciando cliente RPI...")

import os
os.environ['DISPLAY'] = ':0'

import cv2
import paho.mqtt.client as mqtt
import json
import time
import numpy as np
import base64
import threading
import serial
import struct

# GPIO para el relé
try:
    import RPi.GPIO as GPIO
    GPIO_OK = True
except ImportError:
    GPIO_OK = False
    print("WARN: RPi.GPIO no disponible")

print("Librerias cargadas OK", flush=True)

# ==============================================================================
#                      CONFIGURACIÓN DEL RELÉ
# ==============================================================================
RELAY_PIN = 17  # GPIO17 (Pin físico 11) - Cambiar según tu conexión
RELAY_ACTIVE_LOW = True  # True si el relé se activa con LOW (la mayoría de módulos)
RELAY_OPEN_TIME = 4  # Segundos que permanece abierto

# Variables de control para evitar interferencia
relay_is_active = False  # Flag para evitar activaciones múltiples
relay_lock = threading.Lock()  # Lock para sincronización de hilos
last_relay_activation = 0  # Timestamp de última activación
RELAY_COOLDOWN = 5  # Segundos mínimos entre activaciones

def setup_relay():
    """Configurar el pin GPIO para el relé con protecciones"""
    if not GPIO_OK:
        return False
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Configurar pin como salida con pull-up interno para evitar estados flotantes
        GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW)

        # Pequeña pausa para estabilización
        time.sleep(0.1)

        # Asegurar estado inicial desactivado
        if RELAY_ACTIVE_LOW:
            GPIO.output(RELAY_PIN, GPIO.HIGH)  # HIGH = desactivado
        else:
            GPIO.output(RELAY_PIN, GPIO.LOW)   # LOW = desactivado

        print(f"[RELAY] Configurado en GPIO{RELAY_PIN} (Active {'LOW' if RELAY_ACTIVE_LOW else 'HIGH'})")
        return True
    except Exception as e:
        print(f"[RELAY] Error configurando: {e}")
        return False

def activate_relay():
    """Activar el relé para abrir la puerta/cerradura con protecciones"""
    global relay_is_active, last_relay_activation

    if not GPIO_OK:
        print("[RELAY] GPIO no disponible")
        return False

    # Verificar cooldown para evitar activaciones muy seguidas
    current_time = time.time()
    if current_time - last_relay_activation < RELAY_COOLDOWN:
        print(f"[RELAY] Cooldown activo, espere {RELAY_COOLDOWN - (current_time - last_relay_activation):.1f}s")
        return False

    # Usar lock para evitar condiciones de carrera
    if not relay_lock.acquire(blocking=False):
        print("[RELAY] Activación en progreso, ignorando")
        return False

    if relay_is_active:
        relay_lock.release()
        print("[RELAY] Ya está activo, ignorando")
        return False

    def relay_task():
        global relay_is_active, last_relay_activation
        try:
            relay_is_active = True
            last_relay_activation = time.time()
            print(f"[RELAY] Activando por {RELAY_OPEN_TIME} segundos...")

            # Activar relé con pequeña pausa para evitar picos
            time.sleep(0.05)  # 50ms de estabilización
            if RELAY_ACTIVE_LOW:
                GPIO.output(RELAY_PIN, GPIO.LOW)   # LOW = activado
            else:
                GPIO.output(RELAY_PIN, GPIO.HIGH)  # HIGH = activado

            # Esperar el tiempo configurado
            time.sleep(RELAY_OPEN_TIME)

            # Desactivar relé
            if RELAY_ACTIVE_LOW:
                GPIO.output(RELAY_PIN, GPIO.HIGH)  # HIGH = desactivado
            else:
                GPIO.output(RELAY_PIN, GPIO.LOW)   # LOW = desactivado

            time.sleep(0.05)  # 50ms de estabilización
            print("[RELAY] Desactivado")
        except Exception as e:
            print(f"[RELAY] Error: {e}")
            # En caso de error, asegurar que el relé quede desactivado
            try:
                if RELAY_ACTIVE_LOW:
                    GPIO.output(RELAY_PIN, GPIO.HIGH)
                else:
                    GPIO.output(RELAY_PIN, GPIO.LOW)
            except:
                pass
        finally:
            relay_is_active = False
            relay_lock.release()

    # Ejecutar en hilo separado para no bloquear la UI
    threading.Thread(target=relay_task, daemon=True).start()
    return True

def cleanup_relay():
    """Limpiar configuración GPIO al salir"""
    global relay_is_active
    if GPIO_OK:
        try:
            # Asegurar que el relé esté desactivado antes de limpiar
            if RELAY_ACTIVE_LOW:
                GPIO.output(RELAY_PIN, GPIO.HIGH)
            else:
                GPIO.output(RELAY_PIN, GPIO.LOW)
            time.sleep(0.1)
            GPIO.cleanup(RELAY_PIN)
            relay_is_active = False
        except:
            pass

# Pillow para texto UTF-8 con tildes/ñ en la interfaz
from PIL import ImageFont, ImageDraw, Image

# Nota: La codificación UTF-8 y line buffering ya están configurados al inicio del script

# Ruta de fuente TrueType con soporte Unicode (ajusta si es necesario)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def putText_utf8(frame, text, org, font_scale=0.7, color=(255, 255, 255), thickness=2):
    """
    Dibuja texto UTF-8 (con tildes, ñ, etc.) usando Pillow sobre un frame de OpenCV.
    - frame: imagen BGR de OpenCV (np.array)
    - text: str (UTF-8)
    - org: (x, y) posición aproximada de la línea base del texto (como cv2.putText)
    - font_scale: factor de escala "similar" al de OpenCV
    - color: (B, G, R)
    - thickness: grosor aproximado del trazo
    """
    if frame is None:
        return frame

    # Convertir de BGR (OpenCV) a RGB (Pillow)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Ajustar tamaño de fuente a partir de font_scale
    font_size = max(10, int(22 * font_scale))
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        # Fuente por defecto si falla la ruta
        font = ImageFont.load_default()
        font_size = font.size

    x, y = org

    # Color en RGB para Pillow (invertimos BGR → RGB)
    rgb_color = (color[2], color[1], color[0])

    # Usamos stroke_width como aproximación al grosor
    stroke_width = max(1, thickness - 1)

    # Pillow posiciona en la parte superior izquierda, mientras que cv2.putText
    # usa y como línea base. Restamos font_size para aproximar el comportamiento.
    draw.text(
        (x, y - font_size),
        text,
        font=font,
        fill=rgb_color,
        stroke_width=stroke_width,
        stroke_fill=rgb_color
    )

    # Convertir de vuelta a BGR para OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==============================================================================
#                      CLASE AS608 OPTIMIZADA
# ==============================================================================

class AS608:
    """Clase para controlar el sensor de huellas AS608 - OPTIMIZADA"""
    
    # Constantes del protocolo
    STARTCODE = 0xEF01
    DEFAULT_ADDRESS = 0xFFFFFFFF
    DEFAULT_PASSWORD = 0x00000000
    
    # Códigos de paquete
    COMMANDPACKET = 0x01
    DATAPACKET = 0x02
    ACKPACKET = 0x07
    ENDDATAPACKET = 0x08
    
    # Comandos
    GETIMAGE = 0x01
    IMAGE2TZ = 0x02
    MATCH = 0x03
    SEARCH = 0x04
    REGMODEL = 0x05
    STORE = 0x06
    LOAD = 0x07
    DELETE = 0x0C
    EMPTY = 0x0D
    TEMPLATECOUNT = 0x1D
    READTEMPLATEINDEX = 0x1F
    
    def __init__(self, port='/dev/ttyAMA0', baudrate=57600, address=DEFAULT_ADDRESS, password=DEFAULT_PASSWORD):
        """Inicializar la conexión con el sensor"""
        self.address = address
        self.password = password
        
        try:
            self.serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.3
            )
            time.sleep(0.3)
        except serial.SerialException as e:
            raise
    
    def _calculate_checksum(self, data):
        """Calcular el checksum de los datos"""
        return sum(data) & 0xFFFF
    
    def _write_packet(self, packet_type, data):
        """Enviar un paquete al sensor"""
        length = len(data) + 2
        
        packet = struct.pack('>H', self.STARTCODE)
        packet += struct.pack('>I', self.address)
        packet += struct.pack('>B', packet_type)
        packet += struct.pack('>H', length)
        packet += bytes(data)
        
        checksum = self._calculate_checksum(bytes([packet_type]) + struct.pack('>H', length) + bytes(data))
        packet += struct.pack('>H', checksum)
        
        self.serial.write(packet)
        self.serial.flush()
    
    def _read_packet(self, timeout_override=None):
        """Leer un paquete del sensor con timeout ajustable"""
        original_timeout = self.serial.timeout
        if timeout_override is not None:
            self.serial.timeout = timeout_override
        
        try:
            header = self.serial.read(2)
            if len(header) != 2 or struct.unpack('>H', header)[0] != self.STARTCODE:
                return None, None, None
            
            address = self.serial.read(4)
            if len(address) != 4:
                return None, None, None
            
            packet_type_data = self.serial.read(1)
            if len(packet_type_data) != 1:
                return None, None, None
            packet_type = struct.unpack('>B', packet_type_data)[0]
            
            length_data = self.serial.read(2)
            if len(length_data) != 2:
                return None, None, None
            length = struct.unpack('>H', length_data)[0]
            
            data = self.serial.read(length)
            
            if len(data) != length:
                return None, None, None
            
            confirmation = data[0] if len(data) > 0 else None
            
            return packet_type, confirmation, data
        finally:
            self.serial.timeout = original_timeout
    
    def verify_password(self):
        """Verificar la contraseña del sensor"""
        data = [0x13] + list(struct.pack('>I', self.password))
        self._write_packet(self.COMMANDPACKET, data)
        _, confirmation, _ = self._read_packet(timeout_override=0.5)
        
        if confirmation == 0x00:
            print("Sensor AS608 encontrado y contraseña verificada!")
            return True
        else:
            return False
    
    def get_image(self):
        """Capturar imagen de huella"""
        self._write_packet(self.COMMANDPACKET, [self.GETIMAGE])
        _, confirmation, _ = self._read_packet(timeout_override=0.1)
        
        if confirmation == 0x00:
            return True
        elif confirmation == 0x02:
            return False
        else:
            return False
    
    def image_to_tz(self, buffer_id=1):
        """Convertir imagen a template en buffer"""
        self._write_packet(self.COMMANDPACKET, [self.IMAGE2TZ, buffer_id])
        _, confirmation, _ = self._read_packet(timeout_override=1.5)

        if confirmation == 0x00:
            return True
        else:
            return False
    
    def create_model(self):
        """Crear modelo a partir de los buffers 1 y 2"""
        self._write_packet(self.COMMANDPACKET, [self.REGMODEL])
        _, confirmation, _ = self._read_packet(timeout_override=0.5)
        
        if confirmation == 0x00:
            return True
        else:
            return False
    
    def store_model(self, buffer_id=1, location=None):
        """Guardar modelo del buffer en una ubicación"""
        if location is None:
            location = self.get_free_index()
            if location is None:
                return False
        
        data = [self.STORE, buffer_id] + list(struct.pack('>H', location))
        self._write_packet(self.COMMANDPACKET, data)
        _, confirmation, _ = self._read_packet(timeout_override=0.5)
        
        if confirmation == 0x00:
            return location
        else:
            return False
    
    def search(self, buffer_id=1, start_page=0, page_count=200, retries=3):
        """Buscar huella en la base de datos con reintentos"""
        for attempt in range(retries):
            # Limpiar buffer serial antes de buscar
            self.serial.reset_input_buffer()
            time.sleep(0.1)

            data = [self.SEARCH, buffer_id] + list(struct.pack('>H', start_page)) + list(struct.pack('>H', page_count))
            self._write_packet(self.COMMANDPACKET, data)

            original_timeout = self.serial.timeout
            self.serial.timeout = 3.0  # Timeout muy largo para búsqueda

            try:
                header = self.serial.read(2)
                if len(header) != 2 or struct.unpack('>H', header)[0] != self.STARTCODE:
                    if attempt < retries - 1:
                        time.sleep(0.1)
                        continue
                    return None, None
                address = self.serial.read(4)
                packet_type = struct.unpack('>B', self.serial.read(1))[0]
                length = struct.unpack('>H', self.serial.read(2))[0]
                data = self.serial.read(length)

                if len(data) != length:
                    if attempt < retries - 1:
                        time.sleep(0.1)
                        continue
                    return None, None

                confirmation = data[0]

                if confirmation == 0x00:
                    page_id = struct.unpack('>H', data[1:3])[0]
                    score = struct.unpack('>H', data[3:5])[0]
                    return page_id, score
                elif confirmation == 0x09:
                    # Huella no encontrada - no reintentar
                    return None, None
                else:
                    if attempt < retries - 1:
                        time.sleep(0.1)
                        continue
                    return None, None
            finally:
                self.serial.timeout = original_timeout
        return None, None
    
    def delete_model(self, location, count=1):
        """Eliminar modelo(s) de la base de datos"""
        data = [self.DELETE] + list(struct.pack('>H', location)) + list(struct.pack('>H', count))
        self._write_packet(self.COMMANDPACKET, data)
        _, confirmation, _ = self._read_packet(timeout_override=0.5)
        
        if confirmation == 0x00:
            return True
        else:
            return False
    
    def get_template_count(self):
        """Obtener número de templates guardados"""
        self._write_packet(self.COMMANDPACKET, [self.TEMPLATECOUNT])
        
        original_timeout = self.serial.timeout
        self.serial.timeout = 0.5
        
        try:
            header = self.serial.read(2)
            address = self.serial.read(4)
            packet_type = struct.unpack('>B', self.serial.read(1))[0]
            length = struct.unpack('>H', self.serial.read(2))[0]
            data = self.serial.read(length)
            
            confirmation = data[0]
            
            if confirmation == 0x00:
                count = struct.unpack('>H', data[1:3])[0]
                return count
            else:
                return None
        finally:
            self.serial.timeout = original_timeout
    
    def get_free_index(self, start=1, end=200):
        """Encontrar el primer índice libre"""
        for i in range(start, end + 1):
            if not self.check_index_used(i):
                return i
        return None
    
    def check_index_used(self, index):
        """Verificar si un índice está ocupado"""
        data = [self.LOAD, 0x01] + list(struct.pack('>H', index))
        self._write_packet(self.COMMANDPACKET, data)
        _, confirmation, _ = self._read_packet(timeout_override=0.2)
        
        return confirmation == 0x00

    def close(self):
        """Cerrar la conexión serial"""
        if self.serial.is_open:
            self.serial.close()

# ==============================================================================
#                      FIN CLASE AS608
# ==============================================================================


# --- Librerías ---
try:
    import serial
    FINGERPRINT_LIB_OK = True
except ImportError: 
    FINGERPRINT_LIB_OK = False
    print("WARN: No se pudo importar serial.")
except Exception as e: 
    FINGERPRINT_LIB_OK = False
    print(f"Error al importar librerías de huella: {e}")

import os
try:
    # Rutas alternativas para Haar cascade (compatibilidad con diferentes versiones)
    haar_paths = [
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
    ]
    if hasattr(cv2, 'data'):
        haar_paths.insert(0, cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    face_cascade = None
    HAAR_OK = False
    for path in haar_paths:
        if os.path.exists(path):
            face_cascade = cv2.CascadeClassifier(path)
            if not face_cascade.empty():
                print(f"HAAR OK: {path}")
                HAAR_OK = True
                break

    if not HAAR_OK:
        print("ERROR: No se pudo cargar Haar cascade para CARAS.")
except Exception as e:
    HAAR_OK = False
    print(f"Error cargando Haar cascades: {e}")

# ----- CONFIGURACIÓN -----
MQTT_BROKER_IP = "192.168.1.17"  # IP del servidor donde corre Mosquitto
MQTT_PORT = 1883
RPI_CLIENT_ID = "rpi_device_01"
SERIAL_PORT = "/dev/ttyAMA0"
JPEG_QUALITY = 60
STREAM_FPS = 10
FRAME_INTERVAL = 1.0 / STREAM_FPS
CAMERA_INDEX = 0
RESULT_DISPLAY_TIME = 2.0
CAMERA_ZOOM_CROP = 0.0  # 0.0 = sin zoom, 0.2 = 20% de recorte en cada lado

# --- Topics ---
TOPIC_PUB_FACIAL_STREAM = f"acceso/request/facial/stream/{RPI_CLIENT_ID}"
TOPIC_PUB_FACIAL_STOP = f"acceso/request/facial/stop/{RPI_CLIENT_ID}"
TOPIC_PUB_FINGER_REQ = f"acceso/request/fingerprint/{RPI_CLIENT_ID}"
TOPIC_PUB_FACIAL_ENROLL = f"acceso/enroll/facial/data/{RPI_CLIENT_ID}"
TOPIC_PUB_FINGER_ENROLL = f"acceso/enroll/fingerprint/data/{RPI_CLIENT_ID}"
TOPIC_SUB_RESPONSE = f"acceso/response/{RPI_CLIENT_ID}"
TOPIC_SUB_COMMAND = f"acceso/command/{RPI_CLIENT_ID}"

# --- Estados Globales ---
current_state = "IDLE"
display_message = "Seleccione método de acceso"
display_color = (255, 255, 255)
result_end_time = 0
last_frame_sent_time = 0
enroll_user_cedula = None
enroll_user_nombres = None

# --- PIN para salir ---
EXIT_PIN = "1234"
pin_input = ""
pin_error_message = ""
pin_error_time = 0

# Variables de pantalla responsiva
screen_width = 640
screen_height = 480
scale_factor = 1.0

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=RPI_CLIENT_ID)

# --- Sensor de Huella ---
finger = None
SERIAL_PORTS_TO_TRY = ["/dev/ttyAMA0", "/dev/serial0", "/dev/ttyUSB0", "/dev/ttyS0"]
BAUDRATES_TO_TRY = [57600, 9600, 115200]

print("[HUELLA] Buscando sensor AS608...", flush=True)
if FINGERPRINT_LIB_OK:
    for port in SERIAL_PORTS_TO_TRY:
        for baudrate in BAUDRATES_TO_TRY:
            print(f"[HUELLA] Probando {port} @ {baudrate}...", flush=True)
            try:
                finger = AS608(port=port, baudrate=baudrate)
                time.sleep(0.5)
                if finger.verify_password():
                    print(f"[HUELLA] EXITO! Sensor en {port} @ {baudrate}", flush=True)
                    SERIAL_PORT = port
                    break
                else:
                    print(f"[HUELLA] {port} @ {baudrate}: sin respuesta", flush=True)
                    finger.close()
                    finger = None
            except Exception as e:
                if "No such file" not in str(e) and "Permission" not in str(e):
                    print(f"[HUELLA] {port} @ {baudrate}: {e}", flush=True)
                if finger:
                    try:
                        finger.close()
                    except:
                        pass
                finger = None
        if finger:
            break

    if finger is None:
        FINGERPRINT_LIB_OK = False
        print("[HUELLA] ERROR: Sensor AS608 no encontrado", flush=True)
        print("[HUELLA] Verifica: 1) Conexiones TX/RX  2) Alimentacion 3.3V  3) Puerto serial habilitado", flush=True)
    else:
        count = finger.get_template_count()
        print(f"[HUELLA] Huellas almacenadas: {count if count else 0}", flush=True)
else:
    print("[HUELLA] ERROR: Libreria serial no disponible", flush=True)

# ==============================================================================
#                      CLASE BUTTON PARA INTERFAZ TOUCH
# ==============================================================================

class TouchButton:
    """Clase para manejar botones táctiles responsivos"""
    def __init__(self, x, y, w, h, text, color, text_color=(255, 255, 255), icon=None, font_scale_multiplier=1.0):
        # Guardar posiciones base (relativas a 640x480)
        self.base_x = x
        self.base_y = y
        self.base_w = w
        self.base_h = h

        # Posiciones actuales (escaladas)
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.text = text
        self.color = color
        self.text_color = text_color
        self.icon = icon
        self.enabled = True
        self.visible = True
        self.font_scale_multiplier = font_scale_multiplier  # Para texto más grande
    
    def update_scale(self, scale_x, scale_y):
        """Actualizar escala del botón según resolución"""
        self.x = int(self.base_x * scale_x)
        self.y = int(self.base_y * scale_y)
        self.w = int(self.base_w * scale_x)
        self.h = int(self.base_h * scale_y)
    
    def draw(self, frame):
        """Dibujar el botón en el frame"""
        if not self.visible:
            return
        
        # Color más oscuro si está deshabilitado
        color = self.color if self.enabled else tuple(c // 2 for c in self.color)
        
        # Dibujar rectángulo con sombra
        shadow_offset = max(2, int(3 * min(self.w/120, self.h/70)))
        cv2.rectangle(frame, 
                     (self.x + shadow_offset, self.y + shadow_offset), 
                     (self.x + self.w + shadow_offset, self.y + self.h + shadow_offset), 
                     (30, 30, 30), -1)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), color, -1)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (200, 200, 200), 2)
        
        # Escalar tamaños para iconos
        icon_scale = min(self.w / 120, self.h / 70)
        
        # Dibujar icono si existe
        if self.icon:
            icon_y = self.y + int(10 * icon_scale)
            icon_size = int(12 * icon_scale)
            
            if self.icon == "face":
                center = (self.x + self.w // 2, icon_y + icon_size)
                cv2.circle(frame, center, icon_size, self.text_color, max(1, int(2 * icon_scale)))
                cv2.circle(frame, (center[0] - icon_size//3, center[1] - icon_size//4), 
                          max(1, int(2 * icon_scale)), self.text_color, -1)
                cv2.circle(frame, (center[0] + icon_size//3, center[1] - icon_size//4), 
                          max(1, int(2 * icon_scale)), self.text_color, -1)
                cv2.ellipse(frame, (center[0], center[1] + icon_size//3), 
                           (icon_size//2, icon_size//4), 0, 0, 180, self.text_color, 
                           max(1, int(2 * icon_scale)))
                           
            elif self.icon == "finger":
                center = (self.x + self.w // 2, icon_y + icon_size)
                for i in range(3):
                    cv2.ellipse(frame, center, 
                               (int((5 + i*3) * icon_scale), int((9 + i*3) * icon_scale)), 
                               0, 0, 360, self.text_color, max(1, int(icon_scale)))
                               
            elif self.icon == "camera":
                center = (self.x + self.w // 2, icon_y + icon_size)
                cv2.rectangle(frame, 
                             (center[0] - icon_size, center[1] - int(icon_size * 0.7)), 
                             (center[0] + icon_size, center[1] + int(icon_size * 0.7)), 
                             self.text_color, max(1, int(2 * icon_scale)))
                cv2.circle(frame, center, int(icon_size * 0.5), self.text_color, 
                          max(1, int(2 * icon_scale)))
                          
            elif self.icon == "cancel":
                center = (self.x + self.w // 2, icon_y + icon_size)
                cv2.line(frame, 
                        (center[0] - icon_size, center[1] - icon_size), 
                        (center[0] + icon_size, center[1] + icon_size), 
                        self.text_color, max(1, int(2 * icon_scale)))
                cv2.line(frame, 
                        (center[0] - icon_size, center[1] + icon_size), 
                        (center[0] + icon_size, center[1] - icon_size), 
                        self.text_color, max(1, int(2 * icon_scale)))
                        
            elif self.icon == "enroll":
                center = (self.x + self.w // 2, icon_y + icon_size)
                cv2.circle(frame, (center[0] - int(icon_size * 0.4), center[1] - int(icon_size * 0.5)), 
                          int(icon_size * 0.4), self.text_color, max(1, int(2 * icon_scale)))
                cv2.line(frame, 
                        (center[0] - int(icon_size * 0.4), center[1]), 
                        (center[0] - int(icon_size * 0.4), center[1] + int(icon_size * 0.8)), 
                        self.text_color, max(1, int(2 * icon_scale)))
                cv2.line(frame, 
                        (center[0] + int(icon_size * 0.5), center[1]), 
                        (center[0] + int(icon_size * 0.5), center[1] + int(icon_size * 0.6)), 
                        self.text_color, max(1, int(2 * icon_scale)))
                cv2.line(frame, 
                        (center[0] + int(icon_size * 0.2), center[1] + int(icon_size * 0.3)), 
                        (center[0] + int(icon_size * 0.8), center[1] + int(icon_size * 0.3)), 
                        self.text_color, max(1, int(2 * icon_scale)))
        
        # Dibujar texto con escala adaptativa - CENTRADO
        font_scale = 0.5 * min(self.w / 120, self.h / 70) * self.font_scale_multiplier
        thickness = max(1, int(font_scale * 2))

        # Calcular tamaño del texto para centrarlo
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = self.x + (self.w - text_size[0]) // 2

        if self.icon:
            # Si hay icono, texto en la parte inferior del botón
            text_y = self.y + self.h - int(12 * min(self.w/120, self.h/70))
        else:
            # Sin icono, texto centrado verticalmente
            text_y = self.y + (self.h + text_size[1]) // 2

        # Usar Pillow para soportar tildes/ñ en los textos de los botones
        frame[:] = putText_utf8(
            frame,
            self.text,
            (text_x, text_y),
            font_scale=font_scale,
            color=self.text_color,
            thickness=thickness
        )

    def is_clicked(self, x, y):
        """Verificar si el botón fue clickeado"""
        if not self.visible:
            return False
        return (self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h)

# ==============================================================================
#                      FUNCIONES UI CON BOTONES TOUCH
# ==============================================================================

# Botones globales
buttons = {}

def create_buttons():
    """Crear todos los botones de la interfaz - Posiciones base 640x480"""
    global buttons

    # Botones principales (IDLE) - Centrados en la parte inferior
    # Facial a la izquierda, Huella a la derecha
    buttons['facial'] = TouchButton(120, 380, 150, 80, "FACIAL", (0, 150, 0), icon="face")
    buttons['huella'] = TouchButton(370, 380, 150, 80, "HUELLA", (150, 0, 0), icon="finger")

    # Botón enrolar - Centrado debajo (solo visible cuando hay usuario pendiente)
    buttons['enrolar'] = TouchButton(220, 380, 200, 80, "ENROLAR", (0, 150, 150), icon="enroll")
    buttons['enrolar'].visible = False

    # Botón salir - Arriba a la derecha
    buttons['salir'] = TouchButton(550, 10, 80, 40, "SALIR", (180, 0, 0))

    # Botón cancelar - Centrado abajo
    buttons['cancelar'] = TouchButton(240, 390, 160, 60, "CANCELAR", (200, 0, 0), icon="cancel")
    buttons['cancelar'].visible = False

    # Botón capturar foto
    buttons['capturar'] = TouchButton(240, 390, 160, 60, "CAPTURAR", (0, 180, 0), icon="camera")
    buttons['capturar'].visible = False

    # Teclado numérico para PIN (centrado en pantalla base 640x480)
    key_w = 70
    key_h = 55
    key_margin = 8
    key_font = 3.0  # Multiplicador de fuente grande para números

    # Calcular posición centrada
    keypad_total_w = 3 * key_w + 2 * key_margin
    keypad_start_x = (640 - keypad_total_w) // 2  # Centrado horizontal
    keypad_start_y = 140  # Posición vertical

    # Fila 1: 1, 2, 3
    buttons['key_1'] = TouchButton(keypad_start_x, keypad_start_y, key_w, key_h, "1", (70, 70, 70), font_scale_multiplier=key_font)
    buttons['key_2'] = TouchButton(keypad_start_x + key_w + key_margin, keypad_start_y, key_w, key_h, "2", (70, 70, 70), font_scale_multiplier=key_font)
    buttons['key_3'] = TouchButton(keypad_start_x + 2*(key_w + key_margin), keypad_start_y, key_w, key_h, "3", (70, 70, 70), font_scale_multiplier=key_font)

    # Fila 2: 4, 5, 6
    buttons['key_4'] = TouchButton(keypad_start_x, keypad_start_y + key_h + key_margin, key_w, key_h, "4", (70, 70, 70), font_scale_multiplier=key_font)
    buttons['key_5'] = TouchButton(keypad_start_x + key_w + key_margin, keypad_start_y + key_h + key_margin, key_w, key_h, "5", (70, 70, 70), font_scale_multiplier=key_font)
    buttons['key_6'] = TouchButton(keypad_start_x + 2*(key_w + key_margin), keypad_start_y + key_h + key_margin, key_w, key_h, "6", (70, 70, 70), font_scale_multiplier=key_font)

    # Fila 3: 7, 8, 9
    buttons['key_7'] = TouchButton(keypad_start_x, keypad_start_y + 2*(key_h + key_margin), key_w, key_h, "7", (70, 70, 70), font_scale_multiplier=key_font)
    buttons['key_8'] = TouchButton(keypad_start_x + key_w + key_margin, keypad_start_y + 2*(key_h + key_margin), key_w, key_h, "8", (70, 70, 70), font_scale_multiplier=key_font)
    buttons['key_9'] = TouchButton(keypad_start_x + 2*(key_w + key_margin), keypad_start_y + 2*(key_h + key_margin), key_w, key_h, "9", (70, 70, 70), font_scale_multiplier=key_font)

    # Fila 4: Borrar, 0, OK
    buttons['key_del'] = TouchButton(keypad_start_x, keypad_start_y + 3*(key_h + key_margin), key_w, key_h, "<", (180, 100, 0), font_scale_multiplier=key_font)
    buttons['key_0'] = TouchButton(keypad_start_x + key_w + key_margin, keypad_start_y + 3*(key_h + key_margin), key_w, key_h, "0", (70, 70, 70), font_scale_multiplier=key_font)
    buttons['key_ok'] = TouchButton(keypad_start_x + 2*(key_w + key_margin), keypad_start_y + 3*(key_h + key_margin), key_w, key_h, "OK", (0, 150, 0), font_scale_multiplier=2.5)

    # Botón cancelar PIN
    buttons['key_cancel'] = TouchButton(keypad_start_x, keypad_start_y + 4*(key_h + key_margin) + 8, keypad_total_w, 45, "CANCELAR", (180, 0, 0), font_scale_multiplier=1.8)

    # Ocultar teclado por defecto
    for key in ['key_0', 'key_1', 'key_2', 'key_3', 'key_4', 'key_5', 'key_6', 'key_7', 'key_8', 'key_9', 'key_del', 'key_ok', 'key_cancel']:
        buttons[key].visible = False

def update_buttons_scale():
    """Actualizar escala de todos los botones"""
    global buttons, screen_width, screen_height
    scale_x = screen_width / 640.0
    scale_y = screen_height / 480.0

    for button in buttons.values():
        button.update_scale(scale_x, scale_y)

def mouse_callback(event, x, y, flags, param):
    """Callback para eventos del mouse/touch"""
    global current_state, enroll_user_cedula, enroll_user_nombres
    global pin_input, pin_error_message, pin_error_time

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Estado PIN_INPUT - Teclado numérico
    if current_state == "PIN_INPUT":
        # Teclas numéricas - máximo 8 dígitos
        for i in range(10):
            if buttons[f'key_{i}'].is_clicked(x, y):
                if len(pin_input) < 8:
                    pin_input += str(i)
                    # Limpiar mensaje de error al escribir
                    pin_error_message = ""
                return

        # Borrar último dígito
        if buttons['key_del'].is_clicked(x, y):
            if len(pin_input) > 0:
                pin_input = pin_input[:-1]
            pin_error_message = ""
            return

        # Confirmar PIN
        if buttons['key_ok'].is_clicked(x, y):
            if pin_input == EXIT_PIN:
                param['exit'] = True
            else:
                pin_error_message = "Codigo incorrecto"
                pin_error_time = time.time() + 2.0
                pin_input = ""
            return

        # Cancelar entrada de PIN
        if buttons['key_cancel'].is_clicked(x, y):
            pin_input = ""
            pin_error_message = ""
            current_state = "IDLE"
            return
        return

    # Botón SALIR - Muestra teclado PIN
    if buttons['salir'].is_clicked(x, y):
        current_state = "PIN_INPUT"
        pin_input = ""
        pin_error_message = ""
        return

    # Estado IDLE
    if current_state == "IDLE":
        if buttons['facial'].is_clicked(x, y):
            start_facial_verification()
        elif buttons['huella'].is_clicked(x, y):
            capture_and_send_fingerprint_access()
        elif buttons['enrolar'].is_clicked(x, y) and enroll_user_cedula:
            start_admin_enrollment(enroll_user_cedula, enroll_user_nombres)

    # Estado ADMIN_ENROLL_PHOTO
    elif current_state == "ADMIN_ENROLL_PHOTO":
        if buttons['capturar'].is_clicked(x, y):
            admin_enroll_capture_photo(param['current_frame'])
        elif buttons['cancelar'].is_clicked(x, y):
            cancel_operation()

    # Otros estados con cancelar
    elif current_state in ["VERIFYING_FACIAL", "VERIFYING_FINGER", "ADMIN_ENROLL_FINGER"]:
        if buttons['cancelar'].is_clicked(x, y):
            cancel_operation()

def draw_ui(frame, message, color=(255, 255, 255)):
    """Dibujar interfaz de usuario responsiva"""
    global buttons, screen_width, screen_height
    
    if frame is None:
        frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    h, w, _ = frame.shape
    
    # Calcular escalas
    scale_x = w / 640.0
    scale_y = h / 480.0
    font_scale_base = min(scale_x, scale_y)
    
    # Dibujar barra superior con título
    header_height = int(50 * scale_y)
    cv2.rectangle(frame, (0, 0), (w, header_height), (40, 40, 40), -1)
    
    title_font_scale = 0.9 * font_scale_base
    title_thickness = max(1, int(2 * font_scale_base))
    frame = putText_utf8(
        frame,
        "SISTEMA DE ACCESO",
        (int(15 * scale_x), int(32 * scale_y)),
        font_scale=title_font_scale,
        color=(255, 255, 255),
        thickness=title_thickness
    )
    
    # Dibujar mensaje de estado SOLO si NO está en IDLE o tiene información importante
    if current_state != "IDLE" or enroll_user_nombres:
        msg_top = int(60 * scale_y)
        msg_bottom = int(110 * scale_y)
        msg_margin = int(10 * scale_x)
        
        cv2.rectangle(frame, (msg_margin, msg_top), (w - msg_margin, msg_bottom), (50, 50, 50), -1)
        cv2.rectangle(frame, (msg_margin, msg_top), (w - msg_margin, msg_bottom), color, 2)
        
        # Ajustar tamaño del texto según longitud del mensaje
        msg_font_scale = (0.5 if len(message) > 35 else 0.6) * font_scale_base
        msg_thickness = max(1, int(2 * font_scale_base))
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, msg_font_scale, msg_thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (msg_top + msg_bottom + text_size[1]) // 2
        
        frame = putText_utf8(
            frame,
            message,
            (text_x, text_y),
            font_scale=msg_font_scale,
            color=color,
            thickness=msg_thickness
        )
    
    # Configurar visibilidad de botones según estado
    if current_state == "IDLE":
        buttons['facial'].visible = True
        buttons['facial'].enabled = True
        # Mostrar botón huella siempre, pero deshabilitado si no hay sensor
        buttons['huella'].visible = True
        buttons['huella'].enabled = FINGERPRINT_LIB_OK
        buttons['enrolar'].visible = (enroll_user_nombres is not None)
        buttons['cancelar'].visible = False
        buttons['capturar'].visible = False
        buttons['salir'].visible = True
        
    elif current_state == "VERIFYING_FACIAL":
        buttons['facial'].visible = False
        buttons['huella'].visible = False
        buttons['enrolar'].visible = False
        buttons['cancelar'].visible = True
        buttons['capturar'].visible = False
        buttons['salir'].visible = False
        
    elif current_state == "VERIFYING_FINGER":
        buttons['facial'].visible = False
        buttons['huella'].visible = False
        buttons['enrolar'].visible = False
        buttons['cancelar'].visible = True
        buttons['capturar'].visible = False
        buttons['salir'].visible = False
        
    elif current_state == "ADMIN_ENROLL_PHOTO":
        buttons['facial'].visible = False
        buttons['huella'].visible = False
        buttons['enrolar'].visible = False
        buttons['cancelar'].visible = True
        buttons['capturar'].visible = True
        buttons['salir'].visible = False
        
        # Rectángulo guía para la foto - CENTRADO y responsivo
        rect_w = int(300 * scale_x)
        rect_h = int(300 * scale_y)
        rect_x = (w - rect_w) // 2
        rect_y = int(140 * scale_y)
        
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 255), 2)
        guide_text = "Posicione su rostro aquí"
        guide_font_scale = 0.5 * font_scale_base
        guide_thickness = max(1, int(font_scale_base))

        frame = putText_utf8(
            frame,
            guide_text,
            (rect_x + int(20 * scale_x), rect_y - int(10 * scale_y)),
            font_scale=guide_font_scale,
            color=(0, 255, 255),
            thickness=guide_thickness
        )
        
    elif current_state == "ADMIN_ENROLL_FINGER":
        buttons['facial'].visible = False
        buttons['huella'].visible = False
        buttons['enrolar'].visible = False
        buttons['cancelar'].visible = True
        buttons['capturar'].visible = False
        buttons['salir'].visible = False
        
    elif current_state == "SHOW_RESULT":
        buttons['facial'].visible = False
        buttons['huella'].visible = False
        buttons['enrolar'].visible = False
        buttons['cancelar'].visible = False
        buttons['capturar'].visible = False
        buttons['salir'].visible = False

    elif current_state == "PIN_INPUT":
        buttons['facial'].visible = False
        buttons['huella'].visible = False
        buttons['enrolar'].visible = False
        buttons['cancelar'].visible = False
        buttons['capturar'].visible = False
        buttons['salir'].visible = False
        # Mostrar teclado numérico
        for key in ['key_0', 'key_1', 'key_2', 'key_3', 'key_4', 'key_5', 'key_6', 'key_7', 'key_8', 'key_9', 'key_del', 'key_ok', 'key_cancel']:
            buttons[key].visible = True

    # Ocultar teclado si no estamos en PIN_INPUT
    if current_state != "PIN_INPUT":
        for key in ['key_0', 'key_1', 'key_2', 'key_3', 'key_4', 'key_5', 'key_6', 'key_7', 'key_8', 'key_9', 'key_del', 'key_ok', 'key_cancel']:
            buttons[key].visible = False

    # Dibujar botones
    for button in buttons.values():
        button.draw(frame)

    # Dibujar interfaz de PIN si estamos en ese estado
    if current_state == "PIN_INPUT":
        # Fondo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Título centrado
        title = "Ingrese codigo para salir"
        title_font_scale = 0.7 * font_scale_base
        title_thickness = max(1, int(2 * font_scale_base))
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, title_thickness)[0]
        title_x = (w - title_size[0]) // 2
        frame = putText_utf8(frame, title, (title_x, int(50*scale_y)),
                            font_scale=title_font_scale, color=(255, 255, 255), thickness=title_thickness)

        # Mostrar PIN ingresado (con asteriscos) - centrado y más grande
        pin_display = "*" * len(pin_input) if len(pin_input) > 0 else "____"
        pin_font_scale = 1.5 * font_scale_base
        pin_box_w = int(180 * scale_x)
        pin_box_h = int(50 * scale_y)
        pin_x = (w - pin_box_w) // 2
        pin_y = int(75 * scale_y)
        cv2.rectangle(frame, (pin_x, pin_y), (pin_x + pin_box_w, pin_y + pin_box_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (pin_x, pin_y), (pin_x + pin_box_w, pin_y + pin_box_h), (100, 100, 100), 2)
        # Centrar texto del PIN dentro del recuadro
        pin_text_size = cv2.getTextSize(pin_display, cv2.FONT_HERSHEY_SIMPLEX, pin_font_scale, 2)[0]
        pin_text_x = pin_x + (pin_box_w - pin_text_size[0]) // 2
        pin_text_y = pin_y + (pin_box_h + pin_text_size[1]) // 2
        frame = putText_utf8(frame, pin_display, (pin_text_x, pin_text_y),
                            font_scale=pin_font_scale, color=(255, 255, 255), thickness=2)

        # Redibujar botones del teclado encima del overlay
        for key in ['key_0', 'key_1', 'key_2', 'key_3', 'key_4', 'key_5', 'key_6', 'key_7', 'key_8', 'key_9', 'key_del', 'key_ok', 'key_cancel']:
            buttons[key].draw(frame)

        # Mostrar mensaje de error si existe - debajo del botón cancelar, en ROJO
        if pin_error_message and time.time() < pin_error_time:
            error_font_scale = 0.7 * font_scale_base
            error_thickness = max(2, int(2 * font_scale_base))
            error_size = cv2.getTextSize(pin_error_message, cv2.FONT_HERSHEY_SIMPLEX, error_font_scale, error_thickness)[0]
            error_x = (w - error_size[0]) // 2
            error_y = int(460 * scale_y)
            # Fondo oscuro para el mensaje de error
            padding = int(10 * min(scale_x, scale_y))
            cv2.rectangle(frame,
                         (error_x - padding, error_y - error_size[1] - padding),
                         (error_x + error_size[0] + padding, error_y + padding),
                         (40, 40, 40), -1)
            cv2.rectangle(frame,
                         (error_x - padding, error_y - error_size[1] - padding),
                         (error_x + error_size[0] + padding, error_y + padding),
                         (0, 0, 200), 2)
            frame = putText_utf8(frame, pin_error_message, (error_x, error_y),
                                font_scale=error_font_scale, color=(0, 0, 255), thickness=error_thickness)
    
    # Mensaje en IDLE: aparece ENCIMA de los botones
    if current_state == "IDLE" and not enroll_user_nombres:
        # Posición encima de los botones
        msg_y = int(350 * scale_y)
        msg_font_scale = 0.7 * font_scale_base
        msg_thickness = max(1, int(2 * font_scale_base))
        
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, msg_font_scale, msg_thickness)[0]
        text_x = (w - text_size[0]) // 2
        
        # Fondo para el texto
        padding = int(15 * min(scale_x, scale_y))
        cv2.rectangle(frame, 
                     (text_x - padding, msg_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, msg_y + padding),
                     (40, 40, 40), -1)
        
        frame = putText_utf8(
            frame,
            message,
            (text_x, msg_y),
            font_scale=msg_font_scale,
            color=color,
            thickness=msg_thickness
        )
    
    # Info de enrolamiento - Arriba a la izquierda (responsivo)
    if enroll_user_nombres and current_state == "IDLE":
        info_x = int(10 * scale_x)
        info_y = int(120 * scale_y)
        info_w = int(300 * scale_x)
        info_h = int(40 * scale_y)
        
        cv2.rectangle(frame, (info_x, info_y), (info_x + info_w, info_y + info_h), (0, 100, 100), -1)
        cv2.rectangle(frame, (info_x, info_y), (info_x + info_w, info_y + info_h), (0, 200, 200), 2)
        
        text = f"Pendiente: {enroll_user_nombres[:18]}"
        info_font_scale = 0.5 * font_scale_base
        info_thickness = max(1, int(font_scale_base))
        
        frame = putText_utf8(
            frame,
            text,
            (info_x + int(10 * scale_x), info_y + int(25 * scale_y)),
            font_scale=info_font_scale,
            color=(255, 255, 255),
            thickness=info_thickness
        )
    
    return frame

def cancel_operation():
    """Cancelar operación actual"""
    global current_state, enroll_user_cedula, enroll_user_nombres
    
    if current_state == "VERIFYING_FACIAL":
        print("Stream cancelado.")
        try:
            mqtt_client.publish(TOPIC_PUB_FACIAL_STOP, "{}")
        except Exception as e:
            print(f"Error publicando STOP: {e}")
    
    if current_state != "IDLE" and current_state != "SHOW_RESULT":
        print("Operación cancelada.")
        current_state = "IDLE"
        enroll_user_cedula = None
        enroll_user_nombres = None

def set_show_result_state(message, status):
    """Mostrar resultado en pantalla"""
    global current_state, display_message, display_color, result_end_time
    current_state = "SHOW_RESULT"
    display_message = message

    if status == "authenticated" or status == "enroll_ok":
        display_color = (0, 255, 0)
    elif status.startswith("denied"):
        display_color = (0, 0, 255)
    elif status.startswith("enroll"):
        display_color = (0, 255, 255)
    else:
        display_color = (255, 255, 0)

    result_end_time = time.time() + RESULT_DISPLAY_TIME

# ==============================================================================
#                      FUNCIONES DE LÓGICA DE ACCESO
# ==============================================================================

facial_frame_count = 0

def start_facial_verification():
    """Iniciar verificación facial"""
    global current_state, display_message, display_color, last_frame_sent_time, facial_frame_count
    print("[FACIAL] Iniciando verificacion facial...")
    current_state = "VERIFYING_FACIAL"
    display_message = "Iniciando reconocimiento facial..."
    display_color = (0, 255, 255)
    last_frame_sent_time = 0
    facial_frame_count = 0

def stream_facial_frames(frame):
    """Streaming de frames para reconocimiento facial"""
    global current_state, display_message, display_color, last_frame_sent_time, facial_frame_count

    if not HAAR_OK:
        set_show_result_state("Error: Haar Cascades", "denied_error")
        return

    if current_state != "VERIFYING_FACIAL":
        return

    scale_factor = 0.5
    small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor,
                            interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        if not display_message.startswith("Parpadee"):
            display_message = "Buscando rostro..."
            display_color = (0, 255, 255)
        return

    (x, y, w, h) = faces[0]
    x, y, w, h = int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    center = (x + w//2, y + h//2)
    cv2.circle(frame, center, 5, (0, 255, 0), -1)

    current_time = time.time()
    if (current_time - last_frame_sent_time) > FRAME_INTERVAL:
        last_frame_sent_time = current_time
        facial_frame_count += 1
        if facial_frame_count % 10 == 1:
            print(f"[FACIAL] Rostro detectado - Frame #{facial_frame_count} enviado")
        
        # Reducir resolución del frame enviado
        send_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LINEAR)
        _, buffer = cv2.imencode('.jpg', send_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        image_bytes = buffer.tobytes()
        
        try:
            mqtt_client.publish(TOPIC_PUB_FACIAL_STREAM, image_bytes, qos=0)
            if not display_message.startswith("Parpadee"):
                display_message = "Analizando... Mire a la cámara"
                display_color = (0, 255, 0)
        except Exception as e:
            print(f"Error MQTT stream: {e}")
            set_show_result_state("Error de red", "denied_error")

def capture_and_send_fingerprint_access():
    """Capturar y enviar huella dactilar"""
    global current_state, display_message, display_color

    print("[HUELLA] Iniciando verificacion de huella...")

    if not FINGERPRINT_LIB_OK or not finger:
        print("[HUELLA] ERROR: Sensor no disponible")
        set_show_result_state("Sensor de huella no disponible", "denied_error")
        return

    current_state = "VERIFYING_FINGER"
    display_message = "Coloque su dedo en el sensor..."
    display_color = (255, 255, 0)

    def task():
        global current_state, display_message
        try:
            print("[HUELLA] Esperando dedo en el sensor...")
            # Limpiar buffer serial antes de comenzar
            finger.serial.reset_input_buffer()

            max_attempts = 100  # Más intentos
            attempts = 0

            while current_state == "VERIFYING_FINGER" and attempts < max_attempts:
                if finger.get_image():
                    print("[HUELLA] Dedo detectado!")
                    break
                time.sleep(0.05)
                attempts += 1

            if current_state != "VERIFYING_FINGER":
                print("[HUELLA] Operacion cancelada")
                return

            if attempts >= max_attempts:
                print("[HUELLA] Timeout: No se detecto huella")
                set_show_result_state("Tiempo agotado: No se detectó huella", "denied_error")
                return

            display_message = "Procesando huella..."
            print("[HUELLA] Procesando imagen...")

            # Esperar un momento para que el sensor estabilice
            time.sleep(0.1)

            # Reintentar image_to_tz si falla
            success = False
            for retry in range(3):
                if finger.image_to_tz(1):
                    success = True
                    break
                print(f"[HUELLA] Reintento image_to_tz ({retry+1}/3)")
                time.sleep(0.1)

            if not success:
                print("[HUELLA] ERROR: No se pudo procesar la imagen")
                set_show_result_state("Error al procesar huella", "denied_error")
                return

            print("[HUELLA] Buscando huella en base de datos...")
            display_message = "Buscando huella..."

            # La función search ya tiene reintentos incorporados
            page_id, score = finger.search(1, retries=3)

            if page_id is None:
                print("[HUELLA] Huella NO encontrada en base de datos")
                set_show_result_state("Huella no reconocida", "denied_unknown")
                return

            fingerprint_id = page_id
            print(f"[HUELLA] ENCONTRADA! ID: {fingerprint_id}, Score: {score}")

            payload = {"fingerprint_id": fingerprint_id}
            mqtt_client.publish(TOPIC_PUB_FINGER_REQ, json.dumps(payload))
            print(f"[HUELLA] Enviado a servidor para verificar acceso")
            display_message = "Verificando acceso..."

        except Exception as e:
            print(f"Error sensor huella: {e}")
            import traceback
            traceback.print_exc()
            set_show_result_state("Error del sensor", "denied_error")

    threading.Thread(target=task, daemon=True).start()

def start_admin_enrollment(cedula, nombres):
    """Iniciar proceso de enrolamiento administrativo"""
    global current_state, display_message, display_color, enroll_user_cedula, enroll_user_nombres
    enroll_user_cedula = cedula
    enroll_user_nombres = nombres
    current_state = "ADMIN_ENROLL_PHOTO"
    display_message = f"Enrolamiento: {nombres[:25]}"
    display_color = (0, 255, 255)

def admin_enroll_capture_photo(frame):
    """Capturar foto para enrolamiento"""
    global current_state, display_message
    
    if frame is None:
        set_show_result_state("Error: Frame inválido", "denied_error")
        return
    
    print(f"Capturando foto para {enroll_user_cedula}...")
    _, buffer = cv2.imencode('.jpg', frame)
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    payload = {"cedula": enroll_user_cedula, "image_b64": image_b64}
    
    try:
        mqtt_client.publish(TOPIC_PUB_FACIAL_ENROLL, json.dumps(payload))
        display_message = "Foto enviada, esperando confirmación..."
    except Exception as e:
        print(f"Error publicando enrol facial: {e}")
        set_show_result_state("Error de red MQTT", "denied_error")

def admin_enroll_fingerprint():
    """Enrolar huella dactilar"""
    global current_state, display_message, display_color
    
    if not FINGERPRINT_LIB_OK or not finger:
        set_show_result_state("Sensor de huella no disponible", "denied_error")
        return
    
    current_state = "ADMIN_ENROLL_FINGER"
    display_message = f"Enrolando huella: {enroll_user_nombres[:25]}"
    display_color = (0, 255, 255)
    
    def task():
        global current_state, display_message
        try:
            free_slot_id = finger.get_free_index()
            if free_slot_id is None:
                set_show_result_state("Error: Sensor lleno", "denied_error")
                return
            
            for i in range(1, 3):
                display_message = f"Coloque dedo ({i}/2) ID:{free_slot_id}"
                max_attempts = 100
                attempts = 0
                
                while attempts < max_attempts:
                    if finger.get_image():
                        break
                    if current_state != "ADMIN_ENROLL_FINGER":
                        return
                    time.sleep(0.05)
                    attempts += 1
                
                if attempts >= max_attempts:
                    set_show_result_state("Tiempo agotado esperando dedo", "denied_error")
                    return
                
                if not finger.image_to_tz(i):
                    set_show_result_state(f"Error procesar huella ({i}/2)", "denied_error")
                    return
                
                if i == 1:
                    display_message = "Retire el dedo..."
                    time.sleep(0.5)
                    while finger.get_image():
                        if current_state != "ADMIN_ENROLL_FINGER":
                            return
                        time.sleep(0.05)
            
            if not finger.create_model():
                set_show_result_state("Error: Huellas no coinciden", "denied_error")
                return
            
            location_saved = finger.store_model(1, free_slot_id)
            
            if not location_saved:
                set_show_result_state("Error al guardar en sensor", "denied_error")
                return
            
            payload = {"cedula": enroll_user_cedula, "fingerprint_id": location_saved}
            mqtt_client.publish(TOPIC_PUB_FINGER_ENROLL, json.dumps(payload))
            print(f"ID huella {location_saved} para {enroll_user_cedula} enviado.")
            display_message = "Huella enviada. Esperando confirmación..."
            
        except Exception as e:
            print(f"Error en enrolamiento huella: {e}")
            set_show_result_state("Error del sensor", "denied_error")
    
    threading.Thread(target=task, daemon=True).start()

def delete_fingerprint_from_sensor(fingerprint_id):
    """Eliminar huella del sensor"""
    global finger
    if not FINGERPRINT_LIB_OK or not finger:
        return
    print(f"Intentando borrar huella ID {fingerprint_id}...")
    if finger.delete_model(fingerprint_id):
        print("Éxito.")
    else:
        print("Error al borrar.")

# ==============================================================================
#                      FUNCIONES MQTT
# ==============================================================================

def on_connect(client, userdata, flags, reason_code, properties):
    """Callback de conexión MQTT (VERSION2)"""
    if reason_code == 0:
        print(f"Conectado Broker MQTT: {MQTT_BROKER_IP}")
        client.subscribe(TOPIC_SUB_RESPONSE)
        client.subscribe(TOPIC_SUB_COMMAND)
        print(f"Suscrito a {TOPIC_SUB_RESPONSE} y {TOPIC_SUB_COMMAND}")
    else:
        print(f"Falló conexión MQTT: {reason_code}")

def on_message(client, userdata, msg):
    """Callback de mensajes MQTT"""
    global current_state, display_message, display_color, result_end_time
    global enroll_user_cedula, enroll_user_nombres
    
    print(f"MSG Recibido: {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        
        if msg.topic == TOPIC_SUB_RESPONSE:
            status = payload.get("status", "denied_error")
            nombres = payload.get("nombres", "")
            
            if status.startswith("verifying"):
                if current_state == "VERIFYING_FACIAL":
                    display_message = nombres
                    display_color = (0, 255, 255)
                return
            
            elif status == "enroll_facial_ok":
                if current_state == "ADMIN_ENROLL_PHOTO":
                    print("OK foto. Iniciando enrol huella.")
                    current_state = "ADMIN_ENROLL_FINGER"
                    admin_enroll_fingerprint()
                return
            
            elif status == "enroll_finger_ok":
                if current_state == "ADMIN_ENROLL_FINGER":
                    print("OK huella. Enrolamiento completo.")
                    enroll_user_cedula = None
                    enroll_user_nombres = None
                    set_show_result_state("Enrolamiento completo!", "enroll_ok")
                return
            
            elif status.startswith("enroll") and "fail" in status:
                if current_state.startswith("ADMIN_ENROLL"):
                    print(f"Falló enrol: {status}")
                    enroll_user_cedula = None
                    enroll_user_nombres = None
                    set_show_result_state(f"Error Enrol ({status})", "denied_error")
                return
            
            # Resultados finales de acceso
            message = ""
            if status == "authenticated":
                message = f"Bienvenido, {nombres}"
                # Activar el relé para abrir la puerta
                activate_relay()
            elif status == "denied_unknown":
                message = "ACCESO DENEGADO: Desconocido"
            elif status == "denied_no_access":
                message = "ACCESO DENEGADO: Sin permiso"
            elif status == "denied_spoofing":
                message = "ACCESO DENEGADO: SPOOFING"
            elif status == "denied_error" and nombres == "Modelo no entrenado":
                message = "Error: Modelo NO entrenado"
            else:
                message = f"ACCESO DENEGADO: {status}"
            
            set_show_result_state(message, status)
        
        elif msg.topic == TOPIC_SUB_COMMAND:
            command = payload.get("command")
            
            if command == "start_admin_enroll":
                cedula = payload.get("user_cedula")
                nombres = payload.get("user_nombres")
                if cedula and nombres:
                    enroll_user_cedula = cedula
                    enroll_user_nombres = nombres
                    print(f"Recibido comando para enrolar a {nombres} ({cedula})")
                else:
                    print("Error: Comando start_admin_enroll incompleto.")
            
            elif command == "delete_finger":
                delete_fingerprint_from_sensor(payload.get("fingerprint_id"))
    
    except Exception as e:
        print(f"Error procesando MQTT: {e}")
        set_show_result_state("Error de payload", "denied_error")

# ==============================================================================
#                      LOOP PRINCIPAL
# ==============================================================================

def main():
    """Función principal del sistema"""
    global current_state, display_message, display_color, result_end_time
    global enroll_user_cedula, enroll_user_nombres
    global screen_width, screen_height

    print("=" * 50)
    print("   CLIENTE RPI - SISTEMA DE ACCESO BIOMETRICO")
    print("=" * 50)
    print(f"[CONFIG] Broker MQTT: {MQTT_BROKER_IP}:{MQTT_PORT}")
    print(f"[CONFIG] Puerto serial: {SERIAL_PORT}")
    print(f"[CONFIG] Camara index: {CAMERA_INDEX}")
    print(f"[STATUS] Haar Cascade: {'OK' if HAAR_OK else 'ERROR'}")
    print(f"[STATUS] Sensor huella: {'OK' if FINGERPRINT_LIB_OK else 'NO DISPONIBLE'}")

    # Inicializar relé
    relay_ok = setup_relay()
    print(f"[STATUS] Rele GPIO{RELAY_PIN}: {'OK' if relay_ok else 'NO DISPONIBLE'}")
    print("=" * 50)

    cap = None
    exit_flag = {'exit': False, 'current_frame': None}

    # Conectar MQTT
    print(f"[MQTT] Conectando a {MQTT_BROKER_IP}:{MQTT_PORT}...")
    try:
        mqtt_client.on_connect = on_connect
        mqtt_client.on_message = on_message
        mqtt_client.connect(MQTT_BROKER_IP, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print("[MQTT] Conexion iniciada")
    except Exception as e:
        print(f"[MQTT] ERROR: {e}")
        return
    
    # Inicializar cámara
    print(f"Iniciando cámara en índice {CAMERA_INDEX} con backend V4L2...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"WARN: Falló índice {CAMERA_INDEX}. Intentando índice 0...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("ERROR FATAL: No se puede abrir ninguna cámara.")
            mqtt_client.loop_stop()
            cleanup_relay()
            return
    
    # Configuración optimizada de cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    ret, _ = cap.read()
    if not ret:
        print("WARN: No se pudo leer el primer frame.")
        cap.release()
        mqtt_client.loop_stop()
        return
    
    print("Cámara iniciada y lista.")
    
    window_name = "Sistema de Acceso"
    screen_width = 800
    screen_height = 480

    try:
        with open('/sys/class/graphics/fb0/virtual_size', 'r') as f:
            res = f.read().strip().split(',')
            screen_width = int(res[0])
            screen_height = int(res[1])
    except:
        try:
            import subprocess
            result = subprocess.run(['xrandr'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if '*' in line:
                    res = line.split()[0].split('x')
                    screen_width = int(res[0])
                    screen_height = int(res[1])
                    break
        except:
            pass

    print(f"Resolucion pantalla: {screen_width}x{screen_height}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(window_name, 0, 0)
    cv2.resizeWindow(window_name, screen_width, screen_height)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Crear botones y actualizar escala
    create_buttons()
    update_buttons_scale()
    
    cv2.setMouseCallback(window_name, mouse_callback, exit_flag)
    
    print("Cliente RPi iniciado con interfaz touch.")
    active_frame = None

    # Loop principal
    while not exit_flag['exit']:
        frame = None

        # Leer frame de la cámara
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                # Redimensionar frame a la resolución de pantalla
                frame = cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
                active_frame = frame.copy()
                exit_flag['current_frame'] = frame.copy()
            else:
                # Si falla lectura, mantener el último frame válido
                if active_frame is None:
                    active_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        else:
            if active_frame is None:
                active_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        # Lógica de estados - La cámara siempre muestra el frame actual
        if current_state == "IDLE":
            display_message = "Seleccione el método de acceso"
            if enroll_user_nombres:
                display_message = f"Listo para enrolar: {enroll_user_nombres[:22]}"
            display_color = (255, 255, 255)

        elif current_state == "VERIFYING_FACIAL":
            if active_frame is not None:
                stream_facial_frames(active_frame)

        elif current_state == "VERIFYING_FINGER":
            # Mantener cámara activa durante verificación de huella
            pass

        elif current_state == "ADMIN_ENROLL_PHOTO":
            # Mantener cámara activa para capturar foto
            pass

        elif current_state == "ADMIN_ENROLL_FINGER":
            # Mantener cámara activa durante enrolamiento de huella
            pass
        
        elif current_state == "SHOW_RESULT":
            # Timeout automático para volver a IDLE
            if time.time() > result_end_time:
                current_state = "IDLE"
                continue
            if active_frame is None:
                active_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        elif current_state == "PIN_INPUT":
            # Mantener el frame de la cámara como fondo
            pass

        # Dibujar UI
        frame_to_show = active_frame if active_frame is not None else np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        frame_ui = draw_ui(frame_to_show.copy(), display_message, display_color)
        cv2.imshow(window_name, frame_ui)
        
        # Espera mínima para procesar eventos
        cv2.waitKey(1)
    
    # Limpieza final
    if cap is not None and cap.isOpened():
        cap.release()
    # NO cerrar el sensor de huella - debe permanecer encendido siempre
    # if finger and FINGERPRINT_LIB_OK:
    #     finger.close()
    # Limpiar GPIO del relé
    cleanup_relay()
    cv2.destroyAllWindows()
    mqtt_client.loop_stop()
    print("Cliente RPi detenido.")

if __name__ == "__main__":
    main()
