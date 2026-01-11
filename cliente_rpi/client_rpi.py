import cv2
import paho.mqtt.client as mqtt
import json
import time
import numpy as np
import base64
import threading
import sys
import struct
import io

# ================= Serial (PySerial) =================
try:
    import serial
    FINGERPRINT_LIB_OK = True
except ImportError:
    serial = None
    FINGERPRINT_LIB_OK = False
    print("WARN: No se pudo importar pyserial (serial).")
except Exception as e:
    serial = None
    FINGERPRINT_LIB_OK = False
    print(f"WARN: Error importando pyserial: {e}")
# =====================================================

# ================= GPIO (solo Raspberry Pi) =================
try:
    import RPi.GPIO as GPIO
    SERVO_OK = True
except Exception:
    GPIO = None
    SERVO_OK = False
# ============================================================

# Pillow para texto UTF-8 con tildes/ñ en la interfaz
from PIL import ImageFont, ImageDraw, Image

# Configurar codificación UTF-8 para la consola (si aplica)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

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

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_size = max(10, int(22 * font_scale))
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()

    x, y = org
    rgb_color = (color[2], color[1], color[0])
    stroke_width = max(1, thickness - 1)

    draw.text(
        (x, y - font_size),
        text,
        font=font,
        fill=rgb_color,
        stroke_width=stroke_width,
        stroke_fill=rgb_color,
    )

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# --- Haar Cascade para detección de rostros ---
try:
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    HAAR_OK = not face_cascade.empty()
    if not HAAR_OK:
        print("ERROR: No se pudo cargar Haar cascade para CARAS.")
except Exception as e:
    HAAR_OK = False
    print(f"Error cargando Haar cascades: {e}")


# ----- CONFIGURACIÓN -----
MQTT_BROKER_IP = "192.168.137.1"
MQTT_PORT = 1883
RPI_CLIENT_ID = "rpi_device_01"
SERIAL_PORT = "/dev/ttyAMA0"
JPEG_QUALITY = 60
STREAM_FPS = 10
FRAME_INTERVAL = 1.0 / STREAM_FPS
CAMERA_INDEX = 0
RESULT_DISPLAY_TIME = 2.0

# ----- SERVO PUERTA -----
SERVO_GPIO_PIN = 18
SERVO_FREQ_HZ = 50
SERVO_CLOSED_ANGLE = 0
SERVO_OPEN_ANGLE = 90
DOOR_HOLD_OPEN_SEC = 2.5
DOOR_CLOSE_DURATION_SEC = 2.5
DOOR_OPEN_DURATION_SEC = 0.25


# ==============================================================================
#                      SERVO PUERTA (ABRIR/CERRAR)
# ==============================================================================
class DoorServo:
    """
    Control simple de un servo para simular una puerta:
    - abre a 90°
    - espera
    - cierra suave en 2~3s
    """

    def __init__(self, gpio_pin: int, freq_hz: int = 50, start_angle: float = 0.0):
        self.gpio_pin = gpio_pin
        self.freq_hz = freq_hz
        self._pwm = None
        self._current_angle = float(start_angle)

        if not SERVO_OK:
            raise RuntimeError("RPi.GPIO no disponible (no estás en Raspberry o falta librería).")

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.gpio_pin, GPIO.OUT)

        self._pwm = GPIO.PWM(self.gpio_pin, self.freq_hz)
        self._pwm.start(0)

        self.set_angle(self._current_angle)
        time.sleep(0.2)

    def _angle_to_duty(self, angle: float) -> float:
        angle = max(0.0, min(180.0, float(angle)))
        return 2.5 + (angle / 180.0) * 10.0

    def set_angle(self, angle: float):
        if self._pwm is None:
            return
        duty = self._angle_to_duty(angle)
        self._pwm.ChangeDutyCycle(duty)
        self._current_angle = float(angle)

    def move_to(self, target_angle: float, duration_sec: float = 0.0, steps: int = 40):
        target_angle = max(0.0, min(180.0, float(target_angle)))
        start = float(self._current_angle)

        if duration_sec <= 0:
            self.set_angle(target_angle)
            time.sleep(0.15)
            self._pwm.ChangeDutyCycle(0)
            return

        steps = max(5, int(steps))
        dt = float(duration_sec) / steps
        for i in range(1, steps + 1):
            a = start + (target_angle - start) * (i / steps)
            self.set_angle(a)
            time.sleep(dt)

        time.sleep(0.05)
        self._pwm.ChangeDutyCycle(0)

    def cleanup(self):
        try:
            if self._pwm is not None:
                self._pwm.ChangeDutyCycle(0)
                self._pwm.stop()
        finally:
            self._pwm = None
            try:
                GPIO.cleanup(self.gpio_pin)
            except Exception:
                pass


# ==============================================================================
#                      CLASE AS608 OPTIMIZADA
# ==============================================================================
class AS608:
    """Clase para controlar el sensor de huellas AS608 - OPTIMIZADA"""

    STARTCODE = 0xEF01
    DEFAULT_ADDRESS = 0xFFFFFFFF
    DEFAULT_PASSWORD = 0x00000000

    COMMANDPACKET = 0x01
    DATAPACKET = 0x02
    ACKPACKET = 0x07
    ENDDATAPACKET = 0x08

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

    def __init__(self, port="/dev/ttyAMA0", baudrate=57600, address=DEFAULT_ADDRESS, password=DEFAULT_PASSWORD):
        if not FINGERPRINT_LIB_OK or serial is None:
            raise RuntimeError("PySerial no disponible para usar AS608.")

        self.address = address
        self.password = password

        self.serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.3,
        )
        time.sleep(0.3)

    def _calculate_checksum(self, data):
        return sum(data) & 0xFFFF

    def _write_packet(self, packet_type, data):
        length = len(data) + 2
        packet = struct.pack(">H", self.STARTCODE)
        packet += struct.pack(">I", self.address)
        packet += struct.pack(">B", packet_type)
        packet += struct.pack(">H", length)
        packet += bytes(data)

        checksum = self._calculate_checksum(bytes([packet_type]) + struct.pack(">H", length) + bytes(data))
        packet += struct.pack(">H", checksum)

        self.serial.write(packet)
        self.serial.flush()

    def _read_packet(self, timeout_override=None):
        original_timeout = self.serial.timeout
        if timeout_override is not None:
            self.serial.timeout = timeout_override
        try:
            header = self.serial.read(2)
            if len(header) != 2 or struct.unpack(">H", header)[0] != self.STARTCODE:
                return None, None, None

            address = self.serial.read(4)
            if len(address) != 4:
                return None, None, None

            packet_type_data = self.serial.read(1)
            if len(packet_type_data) != 1:
                return None, None, None
            packet_type = struct.unpack(">B", packet_type_data)[0]

            length_data = self.serial.read(2)
            if len(length_data) != 2:
                return None, None, None
            length = struct.unpack(">H", length_data)[0]

            data = self.serial.read(length)
            if len(data) != length:
                return None, None, None

            confirmation = data[0] if len(data) > 0 else None
            return packet_type, confirmation, data
        finally:
            self.serial.timeout = original_timeout

    def verify_password(self):
        data = [0x13] + list(struct.pack(">I", self.password))
        self._write_packet(self.COMMANDPACKET, data)
        _, confirmation, _ = self._read_packet(timeout_override=0.5)
        if confirmation == 0x00:
            print("Sensor AS608 encontrado y contraseña verificada!")
            return True
        return False

    def get_image(self):
        self._write_packet(self.COMMANDPACKET, [self.GETIMAGE])
        _, confirmation, _ = self._read_packet(timeout_override=0.1)
        return confirmation == 0x00

    def image_to_tz(self, buffer_id=1):
        self._write_packet(self.COMMANDPACKET, [self.IMAGE2TZ, buffer_id])
        _, confirmation, _ = self._read_packet(timeout_override=0.5)
        return confirmation == 0x00

    def create_model(self):
        self._write_packet(self.COMMANDPACKET, [self.REGMODEL])
        _, confirmation, _ = self._read_packet(timeout_override=0.5)
        return confirmation == 0x00

    def store_model(self, buffer_id=1, location=None):
        if location is None:
            location = self.get_free_index()
            if location is None:
                return False

        data = [self.STORE, buffer_id] + list(struct.pack(">H", location))
        self._write_packet(self.COMMANDPACKET, data)
        _, confirmation, _ = self._read_packet(timeout_override=0.5)
        return location if confirmation == 0x00 else False

    def search(self, buffer_id=1, start_page=0, page_count=200):
        data = [self.SEARCH, buffer_id] + list(struct.pack(">H", start_page)) + list(struct.pack(">H", page_count))
        self._write_packet(self.COMMANDPACKET, data)

        original_timeout = self.serial.timeout
        self.serial.timeout = 0.5
        try:
            header = self.serial.read(2)
            if len(header) != 2 or struct.unpack(">H", header)[0] != self.STARTCODE:
                return None, None

            _ = self.serial.read(4)  # address
            _ = self.serial.read(1)  # packet_type
            length = struct.unpack(">H", self.serial.read(2))[0]
            data = self.serial.read(length)
            if len(data) != length:
                return None, None

            confirmation = data[0]
            if confirmation == 0x00:
                page_id = struct.unpack(">H", data[1:3])[0]
                score = struct.unpack(">H", data[3:5])[0]
                return page_id, score
            return None, None
        finally:
            self.serial.timeout = original_timeout

    def delete_model(self, location, count=1):
        data = [self.DELETE] + list(struct.pack(">H", location)) + list(struct.pack(">H", count))
        self._write_packet(self.COMMANDPACKET, data)
        _, confirmation, _ = self._read_packet(timeout_override=0.5)
        return confirmation == 0x00

    def get_free_index(self, start=1, end=200):
        for i in range(start, end + 1):
            if not self.check_index_used(i):
                return i
        return None

    def check_index_used(self, index):
        data = [self.LOAD, 0x01] + list(struct.pack(">H", index))
        self._write_packet(self.COMMANDPACKET, data)
        _, confirmation, _ = self._read_packet(timeout_override=0.2)
        return confirmation == 0x00

    def close(self):
        if self.serial and getattr(self.serial, "is_open", False):
            self.serial.close()


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

# Variables de pantalla responsiva
screen_width = 640
screen_height = 480

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=RPI_CLIENT_ID)

# --- Servo Puerta (global) ---
door_servo = None
door_cycle_lock = threading.Lock()
door_cycle_active = False

if SERVO_OK:
    try:
        door_servo = DoorServo(
            gpio_pin=SERVO_GPIO_PIN,
            freq_hz=SERVO_FREQ_HZ,
            start_angle=SERVO_CLOSED_ANGLE,
        )
        print("Servo puerta listo.")
    except Exception as e:
        door_servo = None
        print(f"WARN: No se pudo inicializar el servo: {e}")
else:
    print("INFO: Servo deshabilitado (no es Raspberry / no está RPi.GPIO).")

# --- Sensor de Huella ---
finger = None
if FINGERPRINT_LIB_OK:
    try:
        finger = AS608(port=SERIAL_PORT, baudrate=57600)
        if not finger.verify_password():
            FINGERPRINT_LIB_OK = False
            finger = None
            print("Sensor NO encontrado o contraseña incorrecta.")
    except Exception as e:
        FINGERPRINT_LIB_OK = False
        finger = None
        print(f"Error sensor huella: {e}")


# ==============================================================================
#                      CLASE BUTTON PARA INTERFAZ TOUCH
# ==============================================================================
class TouchButton:
    """Clase para manejar botones táctiles responsivos"""

    def __init__(self, x, y, w, h, text, color, text_color=(255, 255, 255), icon=None):
        self.base_x = x
        self.base_y = y
        self.base_w = w
        self.base_h = h

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

    def update_scale(self, scale_x, scale_y):
        self.x = int(self.base_x * scale_x)
        self.y = int(self.base_y * scale_y)
        self.w = int(self.base_w * scale_x)
        self.h = int(self.base_h * scale_y)

    def draw(self, frame):
        if not self.visible:
            return

        color = self.color if self.enabled else tuple(c // 2 for c in self.color)

        shadow_offset = max(2, int(3 * min(self.w / 120, self.h / 70)))
        cv2.rectangle(
            frame,
            (self.x + shadow_offset, self.y + shadow_offset),
            (self.x + self.w + shadow_offset, self.y + self.h + shadow_offset),
            (30, 30, 30),
            -1,
        )
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), color, -1)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (200, 200, 200), 2)

        icon_scale = min(self.w / 120, self.h / 70)

        if self.icon:
            icon_y = self.y + int(10 * icon_scale)
            icon_size = int(12 * icon_scale)

            if self.icon == "face":
                center = (self.x + self.w // 2, icon_y + icon_size)
                cv2.circle(frame, center, icon_size, self.text_color, max(1, int(2 * icon_scale)))
                cv2.circle(
                    frame,
                    (center[0] - icon_size // 3, center[1] - icon_size // 4),
                    max(1, int(2 * icon_scale)),
                    self.text_color,
                    -1,
                )
                cv2.circle(
                    frame,
                    (center[0] + icon_size // 3, center[1] - icon_size // 4),
                    max(1, int(2 * icon_scale)),
                    self.text_color,
                    -1,
                )
                cv2.ellipse(
                    frame,
                    (center[0], center[1] + icon_size // 3),
                    (icon_size // 2, icon_size // 4),
                    0,
                    0,
                    180,
                    self.text_color,
                    max(1, int(2 * icon_scale)),
                )

            elif self.icon == "finger":
                center = (self.x + self.w // 2, icon_y + icon_size)
                for i in range(3):
                    cv2.ellipse(
                        frame,
                        center,
                        (int((5 + i * 3) * icon_scale), int((9 + i * 3) * icon_scale)),
                        0,
                        0,
                        360,
                        self.text_color,
                        max(1, int(icon_scale)),
                    )

            elif self.icon == "camera":
                center = (self.x + self.w // 2, icon_y + icon_size)
                cv2.rectangle(
                    frame,
                    (center[0] - icon_size, center[1] - int(icon_size * 0.7)),
                    (center[0] + icon_size, center[1] + int(icon_size * 0.7)),
                    self.text_color,
                    max(1, int(2 * icon_scale)),
                )
                cv2.circle(frame, center, int(icon_size * 0.5), self.text_color, max(1, int(2 * icon_scale)))

            elif self.icon == "cancel":
                center = (self.x + self.w // 2, icon_y + icon_size)
                cv2.line(
                    frame,
                    (center[0] - icon_size, center[1] - icon_size),
                    (center[0] + icon_size, center[1] + icon_size),
                    self.text_color,
                    max(1, int(2 * icon_scale)),
                )
                cv2.line(
                    frame,
                    (center[0] - icon_size, center[1] + icon_size),
                    (center[0] + icon_size, center[1] - icon_size),
                    self.text_color,
                    max(1, int(2 * icon_scale)),
                )

            elif self.icon == "enroll":
                center = (self.x + self.w // 2, icon_y + icon_size)
                cv2.circle(
                    frame,
                    (center[0] - int(icon_size * 0.4), center[1] - int(icon_size * 0.5)),
                    int(icon_size * 0.4),
                    self.text_color,
                    max(1, int(2 * icon_scale)),
                )
                cv2.line(
                    frame,
                    (center[0] - int(icon_size * 0.4), center[1]),
                    (center[0] - int(icon_size * 0.4), center[1] + int(icon_size * 0.8)),
                    self.text_color,
                    max(1, int(2 * icon_scale)),
                )
                cv2.line(
                    frame,
                    (center[0] + int(icon_size * 0.5), center[1]),
                    (center[0] + int(icon_size * 0.5), center[1] + int(icon_size * 0.6)),
                    self.text_color,
                    max(1, int(2 * icon_scale)),
                )
                cv2.line(
                    frame,
                    (center[0] + int(icon_size * 0.2), center[1] + int(icon_size * 0.3)),
                    (center[0] + int(icon_size * 0.8), center[1] + int(icon_size * 0.3)),
                    self.text_color,
                    max(1, int(2 * icon_scale)),
                )

        font_scale = 0.45 * min(self.w / 120, self.h / 70)
        thickness = max(1, int(font_scale * 2))
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = self.x + (self.w - text_size[0]) // 2
        text_y = self.y + self.h - int(8 * min(self.w / 120, self.h / 70)) if self.icon else self.y + (self.h + text_size[1]) // 2

        frame[:] = putText_utf8(
            frame,
            self.text,
            (text_x, text_y),
            font_scale=font_scale,
            color=self.text_color,
            thickness=thickness,
        )

    def is_clicked(self, x, y):
        if not self.enabled or not self.visible:
            return False
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h


# ==============================================================================
#                      FUNCIONES UI CON BOTONES TOUCH
# ==============================================================================
buttons = {}


def create_buttons():
    global buttons
    buttons["facial"] = TouchButton(80, 380, 120, 70, "FACIAL", (0, 120, 0), icon="face")
    buttons["huella"] = TouchButton(220, 380, 120, 70, "HUELLA", (120, 0, 0), icon="finger")
    buttons["enrolar"] = TouchButton(360, 380, 120, 70, "ENROLAR", (0, 150, 150), icon="enroll")
    buttons["enrolar"].visible = False

    buttons["salir"] = TouchButton(550, 10, 80, 40, "SALIR", (180, 0, 0))

    buttons["cancelar"] = TouchButton(240, 390, 160, 60, "CANCELAR", (200, 0, 0), icon="cancel")
    buttons["cancelar"].visible = False

    buttons["capturar"] = TouchButton(240, 390, 160, 60, "CAPTURAR", (0, 180, 0), icon="camera")
    buttons["capturar"].visible = False


def update_buttons_scale():
    global buttons, screen_width, screen_height
    scale_x = screen_width / 640.0
    scale_y = screen_height / 480.0
    for button in buttons.values():
        button.update_scale(scale_x, scale_y)


def mouse_callback(event, x, y, flags, param):
    global current_state, enroll_user_cedula, enroll_user_nombres

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if buttons["salir"].is_clicked(x, y):
        param["exit"] = True
        return

    if current_state == "IDLE":
        if buttons["facial"].is_clicked(x, y):
            start_facial_verification()
        elif buttons["huella"].is_clicked(x, y) and FINGERPRINT_LIB_OK:
            capture_and_send_fingerprint_access()
        elif buttons["enrolar"].is_clicked(x, y) and enroll_user_cedula:
            start_admin_enrollment(enroll_user_cedula, enroll_user_nombres)

    elif current_state == "ADMIN_ENROLL_PHOTO":
        if buttons["capturar"].is_clicked(x, y):
            admin_enroll_capture_photo(param["current_frame"])
        elif buttons["cancelar"].is_clicked(x, y):
            cancel_operation()

    elif current_state in ["VERIFYING_FACIAL", "VERIFYING_FINGER", "ADMIN_ENROLL_FINGER"]:
        if buttons["cancelar"].is_clicked(x, y):
            cancel_operation()


def draw_ui(frame, message, color=(255, 255, 255)):
    global buttons, screen_width, screen_height
    if frame is None:
        frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    h, w, _ = frame.shape
    scale_x = w / 640.0
    scale_y = h / 480.0
    font_scale_base = min(scale_x, scale_y)

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
        thickness=title_thickness,
    )

    if current_state != "IDLE" or enroll_user_nombres:
        msg_top = int(60 * scale_y)
        msg_bottom = int(110 * scale_y)
        msg_margin = int(10 * scale_x)

        cv2.rectangle(frame, (msg_margin, msg_top), (w - msg_margin, msg_bottom), (50, 50, 50), -1)
        cv2.rectangle(frame, (msg_margin, msg_top), (w - msg_margin, msg_bottom), color, 2)

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
            thickness=msg_thickness,
        )

    if current_state == "IDLE":
        buttons["facial"].visible = True
        buttons["huella"].visible = FINGERPRINT_LIB_OK
        buttons["enrolar"].visible = (enroll_user_nombres is not None)
        buttons["cancelar"].visible = False
        buttons["capturar"].visible = False
        buttons["salir"].visible = True

    elif current_state == "VERIFYING_FACIAL":
        buttons["facial"].visible = False
        buttons["huella"].visible = False
        buttons["enrolar"].visible = False
        buttons["cancelar"].visible = True
        buttons["capturar"].visible = False
        buttons["salir"].visible = False

    elif current_state == "VERIFYING_FINGER":
        buttons["facial"].visible = False
        buttons["huella"].visible = False
        buttons["enrolar"].visible = False
        buttons["cancelar"].visible = True
        buttons["capturar"].visible = False
        buttons["salir"].visible = False

    elif current_state == "ADMIN_ENROLL_PHOTO":
        buttons["facial"].visible = False
        buttons["huella"].visible = False
        buttons["enrolar"].visible = False
        buttons["cancelar"].visible = True
        buttons["capturar"].visible = True
        buttons["salir"].visible = False

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
            thickness=guide_thickness,
        )

    elif current_state == "ADMIN_ENROLL_FINGER":
        buttons["facial"].visible = False
        buttons["huella"].visible = False
        buttons["enrolar"].visible = False
        buttons["cancelar"].visible = True
        buttons["capturar"].visible = False
        buttons["salir"].visible = False

    elif current_state == "SHOW_RESULT":
        buttons["facial"].visible = False
        buttons["huella"].visible = False
        buttons["enrolar"].visible = False
        buttons["cancelar"].visible = False
        buttons["capturar"].visible = False
        buttons["salir"].visible = False

    for button in buttons.values():
        button.draw(frame)

    if current_state == "IDLE" and not enroll_user_nombres:
        msg_y = int(350 * scale_y)
        msg_font_scale = 0.7 * font_scale_base
        msg_thickness = max(1, int(2 * font_scale_base))

        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, msg_font_scale, msg_thickness)[0]
        text_x = (w - text_size[0]) // 2

        padding = int(15 * min(scale_x, scale_y))
        cv2.rectangle(
            frame,
            (text_x - padding, msg_y - text_size[1] - padding),
            (text_x + text_size[0] + padding, msg_y + padding),
            (40, 40, 40),
            -1,
        )

        frame = putText_utf8(
            frame,
            message,
            (text_x, msg_y),
            font_scale=msg_font_scale,
            color=color,
            thickness=msg_thickness,
        )

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
            thickness=info_thickness,
        )

    return frame


def cancel_operation():
    global current_state, enroll_user_cedula, enroll_user_nombres

    if current_state == "VERIFYING_FACIAL":
        print("Stream cancelado.")
        try:
            mqtt_client.publish(TOPIC_PUB_FACIAL_STOP, "{}")
        except Exception as e:
            print(f"Error publicando STOP: {e}")

    if current_state not in ["IDLE", "SHOW_RESULT"]:
        print("Operación cancelada.")
        current_state = "IDLE"
        enroll_user_cedula = None
        enroll_user_nombres = None


def set_show_result_state(message, status):
    global current_state, display_message, display_color, result_end_time
    current_state = "SHOW_RESULT"
    display_message = message
    if status == "authenticated":
        display_color = (0, 255, 0)
    elif str(status).startswith("denied"):
        display_color = (0, 0, 255)
    elif str(status).startswith("enroll"):
        display_color = (0, 255, 255)
    else:
        display_color = (255, 255, 0)
    result_end_time = time.time() + RESULT_DISPLAY_TIME


# ==============================================================================
#                      FUNCIONES DE LÓGICA DE ACCESO
# ==============================================================================
def start_facial_verification():
    global current_state, display_message, display_color, last_frame_sent_time
    current_state = "VERIFYING_FACIAL"
    display_message = "Iniciando reconocimiento facial..."
    display_color = (0, 255, 255)
    last_frame_sent_time = 0


def stream_facial_frames(frame):
    global current_state, display_message, display_color, last_frame_sent_time

    if not HAAR_OK:
        set_show_result_state("Error: Haar Cascades", "denied_error")
        return

    if current_state != "VERIFYING_FACIAL":
        return

    scale_factor = 0.5
    small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))

    if len(faces) == 0:
        if not display_message.startswith("Parpadee"):
            display_message = "Buscando rostro..."
            display_color = (0, 255, 255)
        return

    (x, y, w, h) = faces[0]
    x, y, w, h = int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    center = (x + w // 2, y + h // 2)
    cv2.circle(frame, center, 5, (0, 255, 0), -1)

    current_time = time.time()
    if (current_time - last_frame_sent_time) > FRAME_INTERVAL:
        last_frame_sent_time = current_time

        send_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LINEAR)
        _, buffer = cv2.imencode(".jpg", send_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
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
    global current_state, display_message, display_color

    if not FINGERPRINT_LIB_OK or not finger:
        set_show_result_state("Sensor de huella no disponible", "denied_error")
        return

    current_state = "VERIFYING_FINGER"
    display_message = "Coloque su dedo en el sensor..."
    display_color = (255, 255, 0)

    def task():
        global current_state, display_message
        try:
            print("Esperando huella para acceso...")
            max_attempts = 50
            attempts = 0

            while current_state == "VERIFYING_FINGER" and attempts < max_attempts:
                if finger.get_image():
                    break
                time.sleep(0.05)
                attempts += 1

            if current_state != "VERIFYING_FINGER":
                return

            if attempts >= max_attempts:
                set_show_result_state("Tiempo agotado: No se detectó huella", "denied_error")
                return

            display_message = "Procesando huella..."

            if not finger.image_to_tz(1):
                set_show_result_state("Error al procesar huella", "denied_error")
                return

            page_id, score = finger.search(1)

            if page_id is None:
                set_show_result_state("Huella no reconocida", "denied_unknown")
                return

            fingerprint_id = page_id
            print(f"Huella encontrada! ID: {fingerprint_id}, Score: {score}")

            payload = {"fingerprint_id": fingerprint_id}
            mqtt_client.publish(TOPIC_PUB_FINGER_REQ, json.dumps(payload))
            display_message = "Verificando acceso..."

        except Exception as e:
            print(f"Error sensor huella: {e}")
            set_show_result_state("Error del sensor", "denied_error")

    threading.Thread(target=task, daemon=True).start()


def start_admin_enrollment(cedula, nombres):
    global current_state, display_message, display_color, enroll_user_cedula, enroll_user_nombres
    enroll_user_cedula = cedula
    enroll_user_nombres = nombres
    current_state = "ADMIN_ENROLL_PHOTO"
    display_message = f"Enrolamiento: {nombres[:25]}"
    display_color = (0, 255, 255)


def admin_enroll_capture_photo(frame):
    global current_state, display_message

    if frame is None:
        set_show_result_state("Error: Frame inválido", "denied_error")
        return

    print(f"Capturando foto para {enroll_user_cedula}...")
    _, buffer = cv2.imencode(".jpg", frame)
    image_b64 = base64.b64encode(buffer).decode("utf-8")
    payload = {"cedula": enroll_user_cedula, "image_b64": image_b64}

    try:
        mqtt_client.publish(TOPIC_PUB_FACIAL_ENROLL, json.dumps(payload))
        display_message = "Foto enviada, esperando confirmación..."
    except Exception as e:
        print(f"Error publicando enrol facial: {e}")
        set_show_result_state("Error de red MQTT", "denied_error")


def admin_enroll_fingerprint():
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
    global finger
    if not FINGERPRINT_LIB_OK or not finger:
        return
    print(f"Intentando borrar huella ID {fingerprint_id}...")
    if finger.delete_model(fingerprint_id):
        print("Éxito.")
    else:
        print("Error al borrar.")


def trigger_door_cycle():
    """
    Abre puerta (90°), espera 2.5s, cierra suave 2~3s.
    Se ejecuta en un hilo y evita ciclos simultáneos.
    """
    global door_cycle_active, door_servo

    if door_servo is None:
        return

    with door_cycle_lock:
        if door_cycle_active:
            return
        door_cycle_active = True

    def task():
        global door_cycle_active
        try:
            door_servo.move_to(SERVO_OPEN_ANGLE, duration_sec=DOOR_OPEN_DURATION_SEC)
            time.sleep(DOOR_HOLD_OPEN_SEC)
            door_servo.move_to(SERVO_CLOSED_ANGLE, duration_sec=DOOR_CLOSE_DURATION_SEC)
        except Exception as e:
            print(f"WARN: Error ciclo puerta: {e}")
        finally:
            with door_cycle_lock:
                door_cycle_active = False

    threading.Thread(target=task, daemon=True).start()


# ==============================================================================
#                      FUNCIONES MQTT
# ==============================================================================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Conectado Broker MQTT: {MQTT_BROKER_IP}")
        client.subscribe(TOPIC_SUB_RESPONSE)
        client.subscribe(TOPIC_SUB_COMMAND)
        print(f"Suscrito a {TOPIC_SUB_RESPONSE} y {TOPIC_SUB_COMMAND}")
    else:
        print(f"Falló conexión MQTT: {rc}")


def on_message(client, userdata, msg):
    global current_state, display_message, display_color, result_end_time
    global enroll_user_cedula, enroll_user_nombres

    print(f"MSG Recibido: {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode("utf-8"))

        if msg.topic == TOPIC_SUB_RESPONSE:
            status = payload.get("status", "denied_error")
            nombres = payload.get("nombres", "")

            if str(status).startswith("verifying"):
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

            elif str(status).startswith("enroll") and "fail" in str(status):
                if str(current_state).startswith("ADMIN_ENROLL"):
                    print(f"Falló enrol: {status}")
                    enroll_user_cedula = None
                    enroll_user_nombres = None
                    set_show_result_state(f"Error Enrol ({status})", "denied_error")
                return

            message = ""
            if status == "authenticated":
                message = f"ACCESO CONCEDIDO: {nombres}"
                trigger_door_cycle()
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
    global current_state, display_message, display_color, result_end_time
    global enroll_user_cedula, enroll_user_nombres
    global screen_width, screen_height

    cap = None
    exit_flag = {"exit": False, "current_frame": None}
    window_name = "Sistema de Acceso - Touch"

    try:
        # Conectar MQTT
        mqtt_client.on_connect = on_connect
        mqtt_client.on_message = on_message
        mqtt_client.connect(MQTT_BROKER_IP, MQTT_PORT, 60)
        mqtt_client.loop_start()

        # Inicializar cámara (backend V4L2 si existe, si no fallback)
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
        print(f"Iniciando cámara en índice {CAMERA_INDEX}...")
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)

        if not cap.isOpened():
            print(f"WARN: Falló índice {CAMERA_INDEX}. Intentando índice 0...")
            cap = cv2.VideoCapture(0, backend)

        if not cap.isOpened():
            print("ERROR FATAL: No se puede abrir ninguna cámara.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, _ = cap.read()
        if not ret:
            print("WARN: No se pudo leer el primer frame.")
            return

        print("Cámara iniciada y lista.")

        # Ventana full screen
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        temp_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imshow(window_name, temp_frame)
        cv2.waitKey(100)

        try:
            x, y, w, h = cv2.getWindowImageRect(window_name)
            if w > 0 and h > 0:
                screen_width = w
                screen_height = h
            else:
                screen_width = 640
                screen_height = 480
        except Exception:
            screen_width = 640
            screen_height = 480

        print(f"Resolución detectada: {screen_width}x{screen_height}")

        create_buttons()
        update_buttons_scale()
        cv2.setMouseCallback(window_name, mouse_callback, exit_flag)

        print("Cliente RPi iniciado con interfaz touch.")
        active_frame = None

        # Loop principal
        while not exit_flag["exit"]:
            frame = None

            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    frame = cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
                    active_frame = frame.copy()
                    exit_flag["current_frame"] = frame.copy()
                else:
                    print("Error leyendo frame.")
                    active_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
                    if current_state not in ["IDLE", "SHOW_RESULT"]:
                        current_state = "IDLE"
            else:
                active_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
                if current_state not in ["IDLE", "SHOW_RESULT"]:
                    print("ERROR: Cámara no disponible. Volviendo a IDLE.")
                    current_state = "IDLE"

            # Lógica de estados
            if current_state == "IDLE":
                display_message = "Seleccione el método de acceso"
                if enroll_user_nombres:
                    display_message = f"Listo para enrolar: {enroll_user_nombres[:22]}"
                display_color = (255, 255, 255)
                if active_frame is None:
                    active_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

            elif current_state == "VERIFYING_FACIAL":
                if active_frame is not None:
                    stream_facial_frames(active_frame)
                else:
                    current_state = "IDLE"

            elif current_state == "VERIFYING_FINGER":
                active_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

            elif current_state == "ADMIN_ENROLL_PHOTO":
                if active_frame is None:
                    current_state = "IDLE"

            elif current_state == "ADMIN_ENROLL_FINGER":
                active_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

            elif current_state == "SHOW_RESULT":
                if time.time() > result_end_time:
                    current_state = "IDLE"
                    continue
                if active_frame is None:
                    active_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

            frame_to_show = active_frame if active_frame is not None else np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            frame_ui = draw_ui(frame_to_show.copy(), display_message, display_color)
            cv2.imshow(window_name, frame_ui)

            cv2.waitKey(1)

    finally:
        # Limpieza final (FUERA del while) + siempre se ejecuta
        try:
            if cap is not None and cap.isOpened():
                cap.release()
        except Exception:
            pass

        try:
            if finger and FINGERPRINT_LIB_OK:
                finger.close()
        except Exception:
            pass

        try:
            if door_servo is not None:
                door_servo.cleanup()
        except Exception as e:
            print(f"WARN: Error cleanup servo: {e}")

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            mqtt_client.loop_stop()
        except Exception:
            pass

        print("Cliente RPi detenido.")


if __name__ == "__main__":
    main()
