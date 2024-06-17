import cv2
from cvzone.PoseModule import PoseDetector

detector = PoseDetector()
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True)
    cv2.imshow("Pose Vision", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
