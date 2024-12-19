import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faseDetection = mpFaceDetection.FaceDetection(0.75)
cTime = 0
pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faseDetection.process(imgRGB)
    if result.detections:
        for id, detection in enumerate(result.detections):
            bboxc = detection.location_data.relative_bounding_box
            ih,iw,ic = img.shape
            bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih) ,\
                    int(bboxc.width * iw),int(bboxc.height * ih)
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img,
                    f"{int(detection.score[0]*100)}",
                    (bbox[0], bbox[1]-30),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 0, 255),
                    2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img,
                f"fps :{(int(fps))}",
                (28, 78),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (255, 0, 255),
                2)

    cv2.imshow("images", img)
    cv2.waitKey(1)
