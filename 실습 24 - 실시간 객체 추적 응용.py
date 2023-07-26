import cv2

cap = cv2.VideoCapture('./data/slow_traffic_small.mp4')

sift = cv2.SIFT_create()

ret, frame = cap.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

prev_keypoints, prev_descriptors = sift.detectAndCompute(prev_gray, None)

ret, frame = cap.read()
x, y, w, h = cv2.selectROI('Select Object', frame, False, False)
track_window = (x, y, w, h)

# 추적기 초기화
roi = prev_gray[y:y+h, x:x+w]
roi_keypoints, roi_descriptors = sift.detectAndCompute(roi, None)
matcher = cv2.BFMatcher(cv2.NORM_L2)
matches = matcher.match(prev_descriptors, roi_descriptors)
matches = sorted(matches, key=lambda x: x.distance)
matching_indices = [m.trainIdx for m in matches]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 추적하기
    roi_gray = gray[y:y+h, x:x+w]
    roi_keypoints, roi_descriptors = sift.detectAndCompute(roi_gray, None)
    matches = matcher.match(prev_descriptors, roi_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    # 추적 결과 그리기
    for match in matches:
        pt1 = prev_keypoints[match.queryIdx].pt
        pt2 = roi_keypoints[match.trainIdx].pt
        x1, y1 = map(int, pt1)
        x2, y2 = map(int, pt2)
        cv2.circle(frame, (x+x2, y+y2), 3, (0,255,0), -1)

    cv2.imshow('Tracking Object',frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    prev_gray = gray.copy()
    prev_keypoints = roi_keypoints
    prev_descriptors = roi_descriptors
    
cap.release()
cv2.destroyAllWindows()