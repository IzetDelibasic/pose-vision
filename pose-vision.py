import cv2
from cvzone.PoseModule import PoseDetector
import time

detector = PoseDetector()
cap = cv2.VideoCapture(0)
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    cv2.imshow("Pose Vision", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
