import json
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

json_dir = './data/anno'
json_paths = glob.glob(os.path.join(json_dir, '*.json'))

label_dict = {'수각류': 0}

for json_path in json_paths:
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    
    image_info = json_data['images']
    annotation_info = json_data['annotations']
    print(image_info)
    print(annotation_info)
    
    file_name = image_info['filename']
    image_id = image_info['id']
    image_width = image_info['width']
    image_height = image_info['height']
    
    print(image_width,image_height)
    
    # 이미지 크기가 3024 4032 정도라 1024 768 로 리사이즈 필요
    new_width = 1024
    new_height = 768

    for ann_info in annotation_info:
        if image_id == ann_info['image_id']:
            
            img_path = os.path.join('./data/images/',file_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 이미지 리사이징
            scale_x = new_width / img.shape[1]  # x축 비율
            scale_y = new_height / img.shape[0]  # y출 비율

            img_resized = cv2.resize(img, (new_width, new_height))

            category_name = ann_info['category_name']
            polygons = ann_info['polygon']

            # 폴리곤 좌표 생성
            points = []
            for polygon in polygons:
                x = polygon['x']
                y = polygon['y']

                # 축소된 사진에서의 포인트 좌표로 옮겨 리스트에 넣기
                resized_x = int(x*scale_x)
                resized_y = int(y*scale_y)

                points.append((resized_x,resized_y))

            cv2.polylines(img_resized, [np.array(points, np.int32).reshape((-1, 1, 2))], isClosed=True, color=(0,255,0), thickness=2)
            
            # 폴리곤 좌표를 이용하여 bbox 만들기
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            cv2.rectangle(img_resized, (x_min, y_min), (x_max,y_max), (255,0,0), 2)

            # 좌표 yolo화
            center_x = ((x_max + x_min) / (2 * new_width))
            center_y = ((y_max + y_min) / (2 * new_height))
            yolo_w = ((x_max - x_min)) / new_width
            yolo_h = ((y_max - y_min)) / new_height

            # 라벨 이름 yolo화
            image_name_temp = file_name.replace('.jpg', '')
            label_num = label_dict[category_name]

        #os.makedirs('./data/anno/yolo_anno', exist_ok=True)
        #with open(f"./data/anno/yolo_anno/{image_name_temp}.txt", "a") as f:
            #f.write(f"{label_num} {center_x} {center_y} {yolo_w} {yolo_h} \n") 
plt.imshow(img_resized)
plt.show()
            