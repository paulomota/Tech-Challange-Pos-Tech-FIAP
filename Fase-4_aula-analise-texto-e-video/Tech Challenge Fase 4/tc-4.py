import cv2;
import face_recognition
import os
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from tqdm import tqdm

VIDEO_PATH = "inputs/video.mp4"
OUTPUT_DIR = "outputs"
FRAMES_TO_ANALYZE = 15  # Intervalo de frames a serem analisados, processa um a cada 30 frames

# Certifique-se de que o diretório de saída exista
os.makedirs(OUTPUT_DIR, exist_ok=True)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def process_video(video_path, output_path):
    print(f"Iniciando processamento do video {video_path}")    
    summary = {"frames": 0, "anomalies": 0, "emotions": {}, "activities": {}}

    # Capturar vídeo do arquivo especificado
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print('Erro ao abrir o video {video_path}')
        return
    
    # Obter propriedades do vídeo
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Total Frames do Video: {total_frames}")
    summary["frames"] = total_frames

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Loop para processar cada frame do vídeo
    count_frames = 0
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = video_capture.read()

        if not ret:
            return
        
        if(count_frames == 0 or count_frames%FRAMES_TO_ANALYZE == 0):
            print(f"Analisando frame numero {count_frames}")
            facial_detection(frame)
            detect_expression(frame, summary)
            detect_activity(frame, summary)

            # Escrever o frame processado no vídeo de saída
            out.write(frame)

        count_frames+=1
    
    # Liberar a captura de vídeo e fechar todas as janelas
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Finalizada etapa de reconhecimento facial")
    return summary

def facial_detection(frame):
    print(f"Iniciando Detecção de faces no frame")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
    face_locations = face_recognition.face_locations(rgb_frame)  # Localizar faces no frame    

    for(top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    


def detect_expression(frame, summary):
    print(f"Detecção de expressões no frame")
    # Analisar o frame para detectar faces e expressões
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    # Iterar sobre cada face detectada
    for face in result:
        # Obter a caixa delimitadora da face
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        
        # Obter a emoção dominante
        dominant_emotion = face['dominant_emotion']        
        
        # Escrever a emoção dominante acima da face
        cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        summary["emotions"][dominant_emotion] = summary["emotions"].get(dominant_emotion, 0) + 1


def detect_activity(frame, summary):
    print(f"Detecção de atividades no frame")

    # Converter para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    activity = ""

    if results.pose_landmarks:
        # Extrair landmarks principais (ombros, quadris, etc.)
        landmarks = results.pose_landmarks.landmark

        # Posição dos ombros e quadris
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Cálculo básico para inferir atividades
        shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_avg_y = (left_hip.y + right_hip.y) / 2
        vertical_diff = shoulder_avg_y - hip_avg_y

        # Verificar alinhamento dos ombros para detectar "deitado"
        shoulder_diff_y = abs(left_shoulder.y - right_shoulder.y)

        if shoulder_diff_y < 0.02:  # Diferença mínima entre ombros
            activity = "Deitado"
        elif vertical_diff > 0.1:
            activity = "Em pé"
        elif vertical_diff < 0.05:
            activity = "Sentado"
        else:
            activity = "Movimento ativo"
        
        summary["activities"][activity] = summary["activities"].get(activity, 0) + 1

        cv2.putText(frame, f'Atividade identificada: {activity}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)


def generate_report(summary):
    report_path = os.path.join(OUTPUT_DIR, "report.txt")

    with open(report_path, "w") as f:
        f.write(f"Frames analisados: {summary["frames"]}\n")
        f.write(f"Anomalias detectadas: {summary["anomalies"]}\n")
        f.write("Emocoes detectadas:\n")
        for emotion, count in summary["emotions"].items():
            f.write(f"  {emotion}: {count}\n")
        f.write("Atividades detectadas:\n")
        for activity, count in summary["activities"].items():
            f.write(f"  {activity}: {count}\n")

    print(f"\nResumo gerado: {report_path}")


if __name__ == "__main__":
    output_video_path = os.path.join(OUTPUT_DIR, 'output_video.mp4')
    summary = process_video(VIDEO_PATH, output_video_path)
    
    generate_report(summary)

    print(f"")