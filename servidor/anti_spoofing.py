from scipy.spatial import distance as dist
# from imutils import face_utils # Ya no es necesario para shape_to_np
import numpy as np
import traceback

# --- Obtener índices de ojos directamente (sin face_utils) ---
FACIAL_LANDMARKS_IDXS = {
    "left_eye": (36, 42),  # Índices para puntos del ojo izquierdo (exclusivo el final)
    "right_eye": (42, 48) # Índices para puntos del ojo derecho (exclusivo el final)
}
# --- Fin de obtención de índices ---\

class BlinkDetector:
    # ==================================================================
    # AJUSTE DE SENSIBILIDAD
    # ==================================================================
    def __init__(self, ear_thresh=0.25, ear_consec_frames=2): # Cambiado de 0.28 a 0.25 (más sensible)
    # ==================================================================
        self.EAR_THRESHOLD = ear_thresh
        self.EAR_CONSEC_FRAMES = ear_consec_frames
        self.frame_counter = 0
        
        # --- Eliminados contadores de estado (blink_counter, liveness_confirmed) ---

        (self.lStart, self.lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]

    def _calculate_ear(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        if C < 1e-6: return self.EAR_THRESHOLD * 2 # Evitar división por cero
        ear = (A + B) / (2.0 * C)
        return ear
        
    def reset(self):
        """ Resetea el contador de frames. app.py se encarga de la lógica de sesión."""
        self.frame_counter = 0

    # ==================================================================
    # LÓGICA DE DETECCIÓN MODIFICADA (STATELESS)
    # ==================================================================
    # Definición con 2 argumentos (gray_frame, landmarks)
    def check_liveness(self, gray_frame, landmarks):
        """
        Comprueba UN parpadeo. Es "stateless".
        Retorna "VIVO" si detecta un parpadeo, y se resetea internamente.
        Retorna un mensaje de estado si no.
        """
        try:
            points = np.array([(p.x, p.y) for p in landmarks.parts()], dtype="int")
            leftEye = points[self.lStart:self.lEnd]
            rightEye = points[self.rStart:self.rEnd]
            leftEAR = self._calculate_ear(leftEye)
            rightEAR = self._calculate_ear(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Comprobar si el ojo está cerrándose
            if ear < self.EAR_THRESHOLD:
                self.frame_counter += 1
                return f"Cerrando ojos..." # Feedback útil
            else:
                # Comprobar si el ojo *acaba* de abrirse tras un parpadeo
                if self.frame_counter >= self.EAR_CONSEC_FRAMES:
                    self.frame_counter = 0 # <-- Auto-reseteo
                    return "VIVO" # <-- ¡Parpadeo detectado!
                
                # Ojos abiertos, sin parpadeo detectado
                self.frame_counter = 0
                return "Mire al frente..." # Estado por defecto

        except Exception as e:
            print(f"[ERROR BlinkDetector] {e}")
            self.frame_counter = 0
            return "Error Liveness"