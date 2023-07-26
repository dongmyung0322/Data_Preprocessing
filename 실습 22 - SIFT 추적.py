import cv2

cap = cv2.VideoCapture('./data/slow_traffic_small.mp4')

# SIFT 객체 생성
sift = cv2.SIFT_create(contrastThreshold=0.02) # 임계값, 임계값이 낮을수록 낮은 대비의 특징점이 검출됨

max_keypoints = 100 # 원하는 최대 특징점 개수

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 특징점 검출
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # 특징점 개수 제한
    if len(keypoints) > max_keypoints:
        keypoints = sorted(keypoints, key=lambda x: -x.response)[:max_keypoints]

    # 시각화
    frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('SIFT', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

