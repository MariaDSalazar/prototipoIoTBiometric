import os
import datetime
import pytz
import cv2
import face_recognition
import dlib
import pickle
import time 
import shutil
import numpy as np
import threading
import paho.mqtt.client as mqtt
import json
import base64
from PIL import Image
import traceback # Para imprimir errores detallados
import random 

# --- IMPORTAR TU SCRIPT DE ANTI-SPOOFING ---
try:
    # Importará la nueva versión con ear_thresh=0.25
    from anti_spoofing import BlinkDetector
    print("Módulo Anti-Spoofing (BlinkDetector) cargado.")
except ImportError:
    print("ERROR: No se encontró el archivo 'anti_spoofing.py'.")
    BlinkDetector = None

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

# --- Configuración ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pickle")
DLIB_PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
ECUADOR_TZ = pytz.timezone('America/Guayaquil')

# ==================================================================
# AJUSTE #5: Reto de 2 parpadeos y Timeout de 12s
# ==================================================================
LIVENESS_TIMEOUT = 12.0 # <-- Aumentado a 12s para dar tiempo a 2 parpadeos

app = Flask(__name__)

# --- Lock para Encodings ---
encoding_lock = threading.Lock() 

app.config['SECRET_KEY'] = 'una-clave-secreta-muy-segura-cambiar-en-prod'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Por favor, inicie sesión para acceder.'
login_manager.login_message_category = 'info'

# --- Cargar Modelos Pesados (Dlib) ---
try:
    face_detector_dlib = dlib.get_frontal_face_detector()
    landmark_predictor_dlib = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
    print("Modelos de Dlib (detector y predictor) cargados.")
except Exception as e:
    print(f"Error al cargar modelos de Dlib: {e}")
    face_detector_dlib = None
    landmark_predictor_dlib = None

# --- GESTOR DE ESTADO DE ANTI-SPOOFING ---
# { 'detector': <BlinkDetector>, 'start_time': <float>, 'blinks_required': <int>, 'blinks_detected': <int> }
client_liveness_info = {} 

# --- Cargar Encodings Faciales ---
known_encodings_data = {"encodings": [], "names": []}
def load_encodings():
    global known_encodings_data
    known_encodings_data = {"encodings": [], "names": []} # Resetear antes de cargar
    if os.path.exists(ENCODINGS_PATH):
        try:
            with encoding_lock:
                with open(ENCODINGS_PATH, 'rb') as f: known_encodings_data = pickle.load(f)
            print(f"Encodings cargados desde '{ENCODINGS_PATH}' ({len(known_encodings_data.get('encodings',[]))} rostros).")
        except Exception as e: print(f"Error al cargar encodings: {e}")
    else: print(f"Advertencia: No se encontró '{ENCODINGS_PATH}'. ¡Necesita re-entrenar!")

load_encodings()

# --- Modelos de BBDD ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    cedula = db.Column(db.String(10), unique=True, nullable=False)
    nombres = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), nullable=False, default='estudiante')
    access_type = db.Column(db.String(20), nullable=False, default='ambos')
    first_login = db.Column(db.Boolean, default=True)
    fingerprint_id = db.Column(db.Integer, unique=True, nullable=True)
    has_facial = db.Column(db.Boolean, default=False)
    has_fingerprint = db.Column(db.Boolean, default=False)

    def set_password(self, password): self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    def check_password(self, password): return bcrypt.check_password_hash(self.password_hash, password)

class AccessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    user_cedula = db.Column(db.String(10), nullable=True)
    user_nombres = db.Column(db.String(100), nullable=True)
    access_type = db.Column(db.String(20), nullable=False, default='desconocido')
    status = db.Column(db.String(50), nullable=False) # Incluirá denied_spoofing

@login_manager.user_loader
def load_user(user_id): return db.session.get(User, int(user_id))

def to_ecuador_time(utc_dt):
    if utc_dt: return utc_dt.replace(tzinfo=pytz.utc).astimezone(ECUADOR_TZ).strftime('%Y-%m-%d %H:%M:%S')
    return 'N/A'

# --- Lógica de Procesamiento Pesado ---
def process_facial_liveness_and_recognition(image_bytes, rpi_client_id):
    global client_liveness_info, landmark_predictor_dlib, known_encodings_data
    
    current_time = time.time()
    info = client_liveness_info.get(rpi_client_id)

    # 1. --- INICIALIZAR ESTADO (SI ES NUEVO) ---
    if info is None:
        if BlinkDetector is None: return "denied_error", "AntiSpoofing no cargado", None
        
        info = {}
        info['detector'] = BlinkDetector() # <-- Crea nueva instancia (con ear_thresh=0.25)
        info['start_time'] = current_time
        # ==========================================================
        # AJUSTE: Reto fijo de 2 parpadeos (SEGURO y USABLE)
        info['blinks_required'] = 2
        # ==========================================================
        info['blinks_detected'] = 0
        client_liveness_info[rpi_client_id] = info
        
        print(f"Nueva prueba de vida para {rpi_client_id}: Se requieren {info['blinks_required']} parpadeos.")

    # 2. --- OBTENER ESTADO ACTUAL ---
    blink_detector = info['detector']
    start_time = info['start_time']
    blinks_required = info['blinks_required']
    blinks_detected = info['blinks_detected']

    try:
        # 3. --- PROCESAR IMAGEN ---
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return "denied_error", "Error decodificando frame", None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Corrección del typo de la versión anterior
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        if not face_detector_dlib or not landmark_predictor_dlib: return "denied_error", "Modelos Dlib no cargados", None
        
        faces_dlib = face_detector_dlib(gray)
        if len(faces_dlib) == 0:
            if blink_detector: blink_detector.reset() # Resetear contador de frames si se pierde cara
            return "verifying_no_face", "Buscando cara...", None

        face = faces_dlib[0]
        landmarks = landmark_predictor_dlib(gray, face)
        
        # 4. --- COMPROBAR ESTADO DE LIVENESS ---
        liveness_status = blink_detector.check_liveness(gray, landmarks)
        elapsed_time = current_time - start_time
        
        # 5. --- MANEJAR TIMEOUT ---
        if elapsed_time > LIVENESS_TIMEOUT:
            print(f"Timeout Liveness para {rpi_client_id} ({blinks_detected}/{blinks_required} parpadeos)")
            if rpi_client_id in client_liveness_info: del client_liveness_info[rpi_client_id]
            return "denied_spoofing", "Timeout Parpadeo", None

        # 6. --- MANEJAR PARPADEO DETECTADO ("VIVO") ---
        if liveness_status == "VIVO":
            blinks_detected += 1
            info['blinks_detected'] = blinks_detected
            print(f"Parpadeo {blinks_detected}/{blinks_required} detectado para {rpi_client_id}!")
            
            # Comprobar si ya se cumplió
            if blinks_detected >= blinks_required:
                 # --- ¡ÉXITO! ---
                print(f"Liveness VIVO ({blinks_required} parpadeos) confirmado!")
                if rpi_client_id in client_liveness_info: del client_liveness_info[rpi_client_id]
                # --- AHORA, CONTINUAR CON RECONOCIMIENTO ---
                pass
            
            else:
                # --- Aún faltan parpadeos ---
                msg = f"Parpadee 1 vez más..."
                return "verifying_liveness", msg, None

        # 7. --- MANEJAR "AÚN VERIFICANDO" (No-VIVO, No-Timeout, No-Exito) ---
        elif blinks_detected < blinks_required:
            # El detector aún no dice "VIVO"
            msg = liveness_status # (ej: "Mire al frente...")
            if blinks_detected == 0:
                 msg = f"Parpadee {blinks_required} veces..."
            elif blinks_detected > 0:
                 msg = f"Parpadee 1 vez más..."
            
            return "verifying_liveness", msg, None
        
        # 8. --- SI SE LLEGA AQUÍ, SIGNIFICA QUE blinks_detected >= blinks_required ---
        # (El código de reconocimiento facial va aquí)

        with encoding_lock:
            encodings_list = list(known_encodings_data.get("encodings", []))
            names_list = list(known_encodings_data.get("names", []))
        
        if not encodings_list:
             print("ERROR CRÍTICO: Modelo no entrenado o vacío. ¡Re-entrene!")
             return "denied_error", "Modelo no entrenado", None

        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        face_locations = [(y, x + w, y + h, x)]
        encodings = face_recognition.face_encodings(rgb, face_locations)
        print(f"Procesando reconocimiento. Encodings detectados: {len(encodings)}")

        if len(encodings) > 0:
            encoding = encodings[0]
            matches = face_recognition.compare_faces(encodings_list, encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(encodings_list, encoding)
            name, cedula = "Desconocido", None
            
            if True in matches:
                 matched_indices = [i for i, match in enumerate(matches) if match]
                 if matched_indices:
                     face_distances_matched = face_distances[matched_indices]
                     best_match_local_index = np.argmin(face_distances_matched)
                     best_match_index = matched_indices[best_match_local_index]
                     name = names_list[best_match_index]
                     cedula = name
                     print(f"Match: {cedula} (Dist: {face_distances[best_match_index]:.4f})")
                 else: print("Error lógico: True in matches pero no indices.")
            else:
                if len(face_distances) > 0:
                    min_distance_index = np.argmin(face_distances)
                    min_distance = face_distances[min_distance_index]
                    closest_cedula = names_list[min_distance_index]
                    print(f"No match (Tolerancia 0.6). Más cercano: {closest_cedula} (Dist: {min_distance:.4f})")
                else: print("No match y no encodings conocidos.")
            
            if cedula != "Desconocido" and cedula is not None:
                user = User.query.filter_by(cedula=cedula).first()
                if user and (user.access_type == 'facial' or user.access_type == 'ambos'): return "authenticated", user.nombres, user.cedula
                else: return "denied_no_access", "Acceso Facial No Permitido", cedula
            else: return "denied_unknown", "Usuario Desconocido", None
        else: return "denied_unknown", "Cara no reconocida (sin encoding)", None
    
    except Exception as e:
        print(f"[Error Procesamiento Facial]\n{traceback.format_exc()}")
        if rpi_client_id in client_liveness_info:
            del client_liveness_info[rpi_client_id]
        return "denied_error", "Error del Servidor", None

def process_fingerprint_recognition(fingerprint_id):
    try:
        if fingerprint_id is None: return "denied_error", "ID de huella nulo", None
        user = User.query.filter_by(fingerprint_id=fingerprint_id).first()
        if user:
            if user.access_type == 'huella' or user.access_type == 'ambos': return "authenticated", user.nombres, user.cedula
            else: return "denied_no_access", "Acceso por Huella No Permitido", user.cedula
        else: return "denied_unknown", "Huella Desconocida", None
    except Exception as e: print(f"[Error Procesamiento Huella] {e}"); return "denied_error", "Error del Servidor", None

# --- Rutas Web (Flask) ---
@app.route('/')
@app.route('/dashboard')
@login_required
def dashboard():
    logs = AccessLog.query.order_by(AccessLog.timestamp.desc()).all()
    return render_template('dashboard.html', title='Dashboard', logs=logs, to_ecuador_time=to_ecuador_time)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']; password = request.form['password']
        user = User.query.filter((User.cedula == username) | (User.cedula == 'admin' and username == 'admin')).first()
        if user and user.check_password(password):
            login_user(user);
            if user.first_login: return redirect(url_for('first_login'))
            return redirect(url_for('dashboard'))
        else: flash('Usuario o contraseña incorrectos.', 'danger')
    return render_template('login.html', title='Login')

@app.route('/first_login', methods=['GET', 'POST'])
@login_required
def first_login():
    if not current_user.first_login: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        new_pass = request.form['new_password']; conf_pass = request.form['confirm_password']
        if new_pass != conf_pass: flash('Las contraseñas no coinciden.', 'danger')
        else:
            current_user.set_password(new_pass); current_user.first_login = False
            db.session.commit(); flash('Contraseña actualizada.', 'success')
            return redirect(url_for('dashboard'))
    return render_template('first_login.html', title='Actualizar Contraseña')

@app.route('/logout')
@login_required
def logout(): logout_user(); return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if current_user.role not in ['admin', 'administrador']: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        cedula = request.form['cedula']; nombres = request.form['nombres']
        role = request.form['role']; access_type = request.form['access_type']
        if User.query.filter_by(cedula=cedula).first():
            flash('La cédula ya existe.', 'danger'); return redirect(url_for('register'))
        new_user = User(cedula=cedula, nombres=nombres, role=role, access_type=access_type, first_login=True)
        new_user.set_password(cedula); db.session.add(new_user); db.session.commit()
        command_payload = {"command": "start_admin_enroll", "user_cedula": cedula, "user_nombres": nombres}
        topic = f"acceso/command/{RPI_CLIENT_ID}"
        try:
            mqtt_client.publish(topic, json.dumps(command_payload))
            flash(f'Usuario {nombres} registrado. RPi notificada para enrolamiento.', 'success')
        except Exception as e:
            print(f"Error MQTT start_admin_enroll: {e}")
            flash(f'Usuario {nombres} registrado, error contactando RPi.', 'warning')
        return redirect(url_for('user_management'))
    return render_template('register.html', title='Registrar Usuario')

@app.route('/users', methods=['GET'])
@login_required
def user_management():
    if current_user.role not in ['admin', 'administrador']: return redirect(url_for('dashboard'))
    users = User.query.filter(User.role != 'admin').order_by(User.nombres).all()
    return render_template('user_management.html', title='Gestionar Usuarios', users=users)

@app.route('/users/update/<int:user_id>', methods=['POST'])
@login_required
def update_user(user_id):
    if current_user.role not in ['admin', 'administrador']: return redirect(url_for('dashboard'))
    user = db.session.get(User, user_id)
    if not user or user.cedula == 'admin': return redirect(url_for('user_management'))
    
    old_access_type = user.access_type; fingerprint_id_to_delete = user.fingerprint_id
    
    user.nombres = request.form['nombres']; user.cedula = request.form['cedula']
    user.role = request.form['role']; user.access_type = request.form['access_type']
    
    if 'reset_password' in request.form:
        user.set_password(user.cedula); user.first_login = True
        flash(f'Contraseña de {user.nombres} reseteada.', 'warning')
    
    trigger_retrain = False 
    
    if (old_access_type == 'huella' or old_access_type == 'ambos') and \
       (user.access_type == 'ninguno' or user.access_type == 'facial') and \
       fingerprint_id_to_delete is not None:
        cmd = {"command": "delete_finger", "fingerprint_id": fingerprint_id_to_delete}
        topic = f"acceso/command/{RPI_CLIENT_ID}"
        try:
            mqtt_client.publish(topic, json.dumps(cmd))
            user.fingerprint_id = None; user.has_fingerprint = False
            flash(f'Huella de {user.nombres} eliminada del sensor.', 'info')
        except Exception as e: print(f"Error publicando comando borrado: {e}")
        
    if (old_access_type == 'facial' or old_access_type == 'ambos') and \
       (user.access_type == 'ninguno' or user.access_type == 'huella'):
        print(f"Acceso facial revocado para {user.nombres}. Se requiere re-entrenamiento.")
        trigger_retrain = True
        
    db.session.commit(); flash(f'Usuario {user.nombres} actualizado.', 'success')
    
    if trigger_retrain:
        print("Iniciando re-entrenamiento completo por revocación de acceso...")
        thread = threading.Thread(target=train_encodings_task); thread.start()
        
    return redirect(url_for('user_management'))

@app.route('/users/delete/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if current_user.role not in ['admin', 'administrador']: return redirect(url_for('dashboard'))
    user = db.session.get(User, user_id)
    if not user or user.cedula == 'admin': return redirect(url_for('user_management'))
    
    fingerprint_id_to_delete = user.fingerprint_id
    user_folder = os.path.join(DATASET_PATH, user.cedula)
    user_had_facial = user.has_facial 
    
    if os.path.exists(user_folder): shutil.rmtree(user_folder)
    db.session.delete(user); db.session.commit()
    
    if fingerprint_id_to_delete is not None:
        cmd = {"command": "delete_finger", "fingerprint_id": fingerprint_id_to_delete}
        topic = f"acceso/command/{RPI_CLIENT_ID}"
        try: mqtt_client.publish(topic, json.dumps(cmd))
        except Exception as e: print(f"Error publicando comando borrado: {e}")
        
    flash(f'Usuario {user.nombres} eliminado.', 'success')
    
    if user_had_facial:
        print("Usuario eliminado tenía datos faciales. Iniciando re-entrenamiento completo...")
        thread = threading.Thread(target=train_encodings_task); thread.start()

    return redirect(url_for('user_management'))

@app.route('/retrain')
@login_required
def retrain_encodings():
    if current_user.role not in ['admin', 'administrador']: return redirect(url_for('dashboard'))
    flash('Iniciando re-entrenamiento (Sincronización Manual)...', 'info') 
    thread = threading.Thread(target=train_encodings_task); thread.start()
    return redirect(url_for('dashboard'))

def train_encodings_task():
    print("Iniciando tarea de (re)entrenamiento COMPLETO...") 
    known_encodings = []; known_names = []
    with app.app_context():
        if not os.path.exists(DATASET_PATH): print(f"ERROR: Directorio '{DATASET_PATH}' no existe."); return
        
        users_in_db_with_facial = {u.cedula for u in User.query.filter(User.access_type.in_(['facial', 'ambos'])).all()}
        
        for cedula_dir in os.listdir(DATASET_PATH):
            user_folder = os.path.join(DATASET_PATH, cedula_dir)
            if os.path.isdir(user_folder):
                
                if cedula_dir not in users_in_db_with_facial:
                    print(f"Omitiendo {cedula_dir} (Usuario no existe o no tiene acceso facial)...");
                    continue

                print(f"Procesando fotos para {cedula_dir}...")
                img_count = 0
                for filename in os.listdir(user_folder):
                    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(user_folder, filename)
                        try:
                            image = face_recognition.load_image_file(img_path)
                            boxes = face_recognition.face_locations(image, model='hog')
                            if len(boxes) > 0:
                                encoding = face_recognition.face_encodings(image, boxes)[0]
                                known_encodings.append(encoding); known_names.append(cedula_dir); img_count += 1
                                print(f"  + {filename}")
                            else: print(f"  - Sin cara: {filename}")
                        except Exception as e: print(f"  ! Error {filename}: {e}")
                
                # Sincronizar 'has_facial' en la BBDD
                user = User.query.filter_by(cedula=cedula_dir).first()
                if user:
                    if img_count > 0 and not user.has_facial:
                        user.has_facial = True;
                        print(f"  -> {img_count} fotos ok. (Actualizando BBDD)")
                    elif img_count == 0 and user.has_facial:
                        user.has_facial = False;
                        print(f"  -> 0 fotos ok. (Actualizando BBDD)")
                    elif img_count > 0:
                        print(f"  -> {img_count} fotos ok. (BBDD ya correcta)")
                    else:
                        print(f"  -> 0 fotos ok. (BBDD ya correcta)")
        
        db.session.commit() # Commit de todos los cambios de 'has_facial'

    if not known_encodings: 
        print("ADVERTENCIA: No se generaron encodings. El archivo .pickle será vaciado.");
        data = {"encodings": [], "names": []}
    else:
        data = {"encodings": known_encodings, "names": known_names}

    try:
        with encoding_lock:
            with open(ENCODINGS_PATH, "wb") as f: pickle.dump(data, f)
        
        print(f"Entrenamiento finalizado. '{ENCODINGS_PATH}' actualizado ({len(known_encodings)} encodings)."); 
        
        # Recargar los encodings en memoria (ya está protegido por el lock)
        load_encodings()
        
    except Exception as e: print(f"Error al guardar {ENCODINGS_PATH}: {e}")


def update_model_with_image(cedula, image_bytes):
    """
    Procesa UNA imagen y la añade de forma incremental al archivo de encodings.
    Es 'thread-safe' usando encoding_lock.
    """
    global known_encodings_data
    print(f"Actualización incremental: Procesando imagen para {cedula}...")
    
    try:
        # 1. Procesar la imagen
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: 
            print("Error: No se pudo decodificar la imagen para el encoding.")
            return False
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_image, model='hog')
        
        if len(boxes) == 0:
            print("Advertencia: No se detectó cara en la imagen de enrolamiento.")
            return False 

        new_encoding = face_recognition.face_encodings(rgb_image, boxes)[0]

        # 2. Actualizar el archivo y la variable global (con bloqueo)
        with encoding_lock:
            print("Lock adquirido. Actualizando encodings...")
            
            # Cargar los datos FRESCOS del disco, por si acaso
            current_data = {"encodings": [], "names": []}
            if os.path.exists(ENCODINGS_PATH):
                try:
                    with open(ENCODINGS_PATH, 'rb') as f:
                        current_data = pickle.load(f)
                except Exception as e:
                    print(f"Error cargando pickle para actualizar: {e}. Se creará uno nuevo.")

            # 3. Añadir nuevo encoding
            current_data["encodings"].append(new_encoding)
            current_data["names"].append(cedula)

            # 4. Guardar de nuevo en el disco
            try:
                with open(ENCODINGS_PATH, "wb") as f:
                    pickle.dump(current_data, f)
                
                # 5. Actualizar la variable global
                known_encodings_data = current_data
                print(f"Encoding para {cedula} añadido. Total: {len(known_encodings_data['encodings'])}")
                return True
                
            except Exception as e:
                print(f"Error CRÍTICO guardando {ENCODINGS_PATH} actualizado: {e}")
                return False
        
    except Exception as e:
        print(f"[Error en update_model_with_image]\n{traceback.format_exc()}")
        return False


@app.cli.command('init-db')
def init_db_command():
    """
    Crea las tablas de la BBDD, el usuario admin, 
    Y ENVÍA UN COMANDO DE LIMPIEZA TOTAL AL SENSOR DE HUELLAS.
    """
    db.create_all()
    if not User.query.filter_by(cedula='admin').first():
        admin = User(cedula='admin', nombres='Admin', role='administrador', first_login=False)
        admin.set_password('admin'); db.session.add(admin); db.session.commit(); print("Usuario admin creado.")
    if not os.path.exists(DATASET_PATH): os.makedirs(DATASET_PATH); print(f"Directorio '{DATASET_PATH}' creado.")
    print("Base de datos inicializada.")

    print("Intentando contactar al Broker MQTT para limpiar sensor RPi...")
    try:
        # Necesitamos un cliente temporal solo para este comando CLI
        cli_mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="flask_cli_init_db") # <-- Actualizado a V2
        
        # Establecer un timeout de conexión corto
        cli_mqtt.connect(MQTT_BROKER_IP, MQTT_PORT, 10)
        
        cmd = {"command": "clear_all_fingers"}
        topic = f"acceso/command/{RPI_CLIENT_ID}"
        
        cli_mqtt.loop_start() # Iniciar loop en background
        
        # Publicar el comando
        result = cli_mqtt.publish(topic, json.dumps(cmd), qos=1) # Usar QoS 1
        result.wait_for_publish(timeout=5) # Esperar a que se publique
        
        print(f"Comando 'clear_all_fingers' publicado en {topic}.")
        
        cli_mqtt.loop_stop()
        cli_mqtt.disconnect()
        print("ÉXITO: Sensor RPi notificado para limpieza total.")
        
    except Exception as e:
        print("---------------------------------------------------------")
        print(f"ADVERTENCIA: No se pudo contactar al Broker MQTT.")
        print(f"Error: {e}")
        print("El sensor de huellas NO FUE LIMPIADO. Si la RPi está")
        print("conectada, el sensor seguirá desincronizado.")
        print("Asegúrese que el broker MQTT y la RPi estén encendidos")
        print("antes de ejecutar 'flask init-db'.")
        print("---------------------------------------------------------")

# --- LÓGICA DE MQTT ---
MQTT_BROKER_IP = "127.0.0.1"; MQTT_PORT = 1883; RPI_CLIENT_ID = "rpi_device_01"
TOPIC_REQ_FACIAL_STREAM = "acceso/request/facial/stream"; TOPIC_REQ_FACIAL_STOP = "acceso/request/facial/stop"
TOPIC_REQ_FINGER = "acceso/request/fingerprint"; TOPIC_ENROLL_FACIAL = "acceso/enroll/facial/data"
TOPIC_ENROLL_FINGER = "acceso/enroll/fingerprint/data"; TOPIC_RESPONSE_BASE = "acceso/response"
TOPIC_COMMAND_BASE = "acceso/command"

# ==========================================================
# CORRECCIÓN DE ERROR (API V2)
# ==========================================================
def on_connect(client, userdata, flags, reason_code, properties): # <-- 5 argumentos
    if reason_code == 0: # <-- Comprobar reason_code
        print(f"Conectado al Broker MQTT en {MQTT_BROKER_IP}!")
        client.subscribe(f"{TOPIC_REQ_FACIAL_STREAM}/#"); client.subscribe(f"{TOPIC_REQ_FACIAL_STOP}/#")
        client.subscribe(f"{TOPIC_REQ_FINGER}/#"); client.subscribe(f"{TOPIC_ENROLL_FACIAL}/#")
        client.subscribe(f"{TOPIC_ENROLL_FINGER}/#"); print(f"Suscrito a topics.")
    else: print(f"Fallo al conectar a MQTT, código {reason_code}")

def on_message(client, userdata, msg): # <-- Esta firma (3 args) es correcta para V2
    global client_liveness_info
    with app.app_context():
        try:
            topic_parts = msg.topic.split('/'); rpi_client_id = topic_parts[-1]
            response_topic = f"{TOPIC_RESPONSE_BASE}/{rpi_client_id}"

            if msg.topic.startswith(TOPIC_REQ_FACIAL_STREAM):
                image_bytes = msg.payload
                status, nombres, cedula = process_facial_liveness_and_recognition(image_bytes, rpi_client_id)
                response_payload = {"status": status, "nombres": nombres}
                if status.startswith("verifying"): client.publish(response_topic, json.dumps(response_payload))
                else:
                    client.publish(response_topic, json.dumps(response_payload))
                    if rpi_client_id in client_liveness_info: del client_liveness_info[rpi_client_id]
                    if status != "denied_error":
                        log = AccessLog(user_cedula=cedula, user_nombres=nombres, access_type='facial', status=status)
                        db.session.add(log); db.session.commit()
                    print(f"Respuesta facial enviada: {response_payload}")

            elif msg.topic.startswith(TOPIC_REQ_FACIAL_STOP):
                print(f"RPi {rpi_client_id} detuvo stream.");
                if rpi_client_id in client_liveness_info: del client_liveness_info[rpi_client_id]

            elif msg.topic.startswith(TOPIC_REQ_FINGER):
                data = json.loads(msg.payload.decode('utf-8')); fingerprint_id = data.get('fingerprint_id')
                status, nombres, cedula = process_fingerprint_recognition(fingerprint_id)
                response_payload = {"status": status, "nombres": nombres}
                client.publish(response_topic, json.dumps(response_payload))
                if status != "denied_error":
                    log = AccessLog(user_cedula=cedula, user_nombres=nombres, access_type='huella', status=status)
                    db.session.add(log); db.session.commit()
                print(f"Respuesta huella enviada: {response_payload}")

            elif msg.topic.startswith(TOPIC_ENROLL_FACIAL):
                data = json.loads(msg.payload.decode('utf-8')); cedula = data.get('cedula'); image_b64 = data.get('image_b64')
                user = User.query.filter_by(cedula=cedula).first();
                if not user: return
                print(f"Recibida foto de enrolamiento para {cedula}...")
                try:
                    image_bytes = base64.b64decode(image_b64)
                    
                    # --- 1. Guardar la foto en el dataset (como antes) ---
                    user_folder = os.path.join(DATASET_PATH, cedula)
                    if not os.path.exists(user_folder): os.makedirs(user_folder)
                    existing_files = len([name for name in os.listdir(user_folder) if os.path.isfile(os.path.join(user_folder, name))])
                    img_name = f"enroll_{existing_files + 1}.jpg"; img_path = os.path.join(user_folder, img_name)
                    with open(img_path, 'wb') as f: f.write(image_bytes)
                    print(f"Foto guardada: {img_path}")

                    # --- 2. Actualizar el modelo (NUEVA LÓGICA) ---
                    # Usar un thread para no bloquear el listener MQTT
                    img_bytes_copy = bytes(image_bytes) # Copiar bytes
                    thread = threading.Thread(target=update_model_with_image, args=(cedula, img_bytes_copy))
                    thread.start()
                    
                    # --- 3. Actualizar BBDD y responder (como antes) ---
                    if not user.has_facial: # Solo actualizar BBDD si era False
                        user.has_facial = True; db.session.commit()
                    
                    client.publish(response_topic, json.dumps({"status": "enroll_facial_ok"}))
                    
                except Exception as e: 
                    print(f"Error guardando foto o iniciando thread de encoding: {e}")
                    print(traceback.format_exc()) 
                    client.publish(response_topic, json.dumps({"status": "enroll_facial_fail"}))

            elif msg.topic.startswith(TOPIC_ENROLL_FINGER):
                data = json.loads(msg.payload.decode('utf-8')); cedula = data.get('cedula'); fingerprint_id = data.get('fingerprint_id')
                user = User.query.filter_by(cedula=cedula).first(); command_topic = f"{TOPIC_COMMAND_BASE}/{rpi_client_id}"
                if not user:
                    cmd = {"command": "delete_finger", "fingerprint_id": fingerprint_id}; client.publish(command_topic, json.dumps(cmd))
                    print(f"Cédula {cedula} no encontrada. Enviando borrado ID {fingerprint_id}."); return
                print(f"Recibido ID de huella {fingerprint_id} para {cedula}...")
                try:
                    existing_user_with_id = User.query.filter_by(fingerprint_id=fingerprint_id).first()
                    if existing_user_with_id and existing_user_with_id.id != user.id:
                         raise Exception(f"ID huella {fingerprint_id} ya en uso por {existing_user_with_id.cedula}")
                    user.fingerprint_id = fingerprint_id; user.has_fingerprint = True; db.session.commit()
                    print(f"ID huella {fingerprint_id} para {cedula} guardado."); client.publish(response_topic, json.dumps({"status": "enroll_finger_ok"}))
                except Exception as e:
                    print(f"Error BBDD guardando ID huella: {e}"); db.session.rollback()
                    client.publish(response_topic, json.dumps({"status": "enroll_finger_fail_db"}))
                    cmd = {"command": "delete_finger", "fingerprint_id": fingerprint_id}; client.publish(command_topic, json.dumps(cmd))
                    print(f"Error BBDD. Enviando borrado ID {fingerprint_id}.")

        except Exception as e: 
            print(f"Error grave en on_message: {e}")
            print(traceback.format_exc()) 

def start_mqtt_listener():
    global mqtt_client
    # Corregido para V2
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="servidor_flask_app") 
    mqtt_client.on_connect = on_connect; mqtt_client.on_message = on_message
    try:
        mqtt_client.connect(MQTT_BROKER_IP, MQTT_PORT, 60); mqtt_client.loop_start()
    except Exception as e: print(f"No se pudo conectar al broker MQTT: {e}")

if __name__ == '__main__':
    if not os.path.exists(DLIB_PREDICTOR_PATH) or BlinkDetector is None:
        print(f"ERROR: No se encuentra '{DLIB_PREDICTOR_PATH}' o 'anti_spoofing.py'")
    else:
        start_mqtt_listener()
        app.run(host='0.0.0.0', port=5000, debug=False)