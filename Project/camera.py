import cv2
import mediapipe as mp
import numpy as np
from style_mediapipe import  draw_styled_landmarks,extract_keypoints
from Transformers_Landmark import get_model,predict
model = get_model()
model.load_weights('./Project/weight/model.h5')
# Khởi tạo Holistic
cap = cv2.VideoCapture(0)
sequence =  []
res = ''
n_frame = 0
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        # Chuyển đổi frame sang định dạng màu BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Dùng Holistic để xử lý frame
        results = holistic.process(frame_rgb)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence_arr = np.array(sequence)
        if len(sequence) == 128:
            res  = predict(model,sequence_arr)
            sequence=[]
            n_frame = 0
        # Vẽ các landmark trên cơ thể và khuôn mặt
        draw_styled_landmarks(frame,results)
        cv2.putText(frame, ' '.join(res), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Holistic', frame)
        n_frame +=1
        print(n_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
print(res)
cap.release()
cv2.destroyAllWindows()