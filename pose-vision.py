import cv2
from cvzone.PoseModule import PoseDetector
import time

detector = PoseDetector()
cap = cv2.VideoCapture(0)
pTime = 0
is_camera_on = True  

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    
    if bboxInfo:
        x, y, w, h = bboxInfo["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    cv2.imshow("Pose Vision", img)
    
    key = cv2.waitKey(1)
    if key == ord('s'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"pose_snapshot_{timestamp}.png"
        cv2.imwrite(filename, img)
        print(f"Snapshot saved as {filename}")
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
