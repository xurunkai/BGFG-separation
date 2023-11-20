import time
from ultralytics import YOLO
import cv2
import numpy as np


class image_preprocess:

    def __init__(self):
        self.model = YOLO("yolov8s.pt")
        args = {'task': 'detect', 'data': 'coco.yaml', 'imgsz': 32, 'single_cls': True, 'model': 'yolov8s.pt',
                'conf': 0.3, 'save': False, 'show': False, 'mode': 'predict'}
        self.model.predictor = self.model._smart_load('predictor')(overrides=args, _callbacks=self.model.callbacks)
        self.model.predictor.setup_model(model=self.model.model, verbose=False)
        # self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(history=600, dist2Threshold=300, detectShadows=False)
        self.kernel = np.ones((3, 3), np.uint8)

    # 畸变矫正
    def distortion_correction(self, image):
        image = cv2.resize(image, (960, 540))
        mat_intri = np.array([[603.95326447, 0, 481.36708033],
                              [0, 604.51242483, 272.66842792],
                              [0, 0, 1]])
        coff_dis = np.array([-3.89127437e-01, 1.92954238e-01, -3.28167575e-04, -2.74500971e-04, -5.41705555e-02])
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_intri, coff_dis, (960, 540), 0, (960, 540))
        dis_image = cv2.undistort(image, mat_intri, coff_dis, None, newcameramtx)
        return dis_image

    def rectangle_filter(self, contours):
        filtered_rects = []
        centers = []
        ignore = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w * h <= 100:
                ignore.append(i)
                continue
            is_contained = False
            for j, other_contour in enumerate(contours):
                if j != i and j not in ignore:
                    other_x, other_y, other_w, other_h = cv2.boundingRect(other_contour)
                    if x > other_x and y > other_y and x + w < other_x + other_w and y + h < other_y + other_h:
                        is_contained = True
                        break
            if not is_contained:
                filtered_rects.append([x, y, x + w, y + h])
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                centers.append([center_x, center_y])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(len(centers) - 1), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255),
                            1, cv2.LINE_AA)
                # 保存中心点到list2中
        filtered_rects = np.array(filtered_rects)
        return filtered_rects, centers

    def rect_fuse(self, contours, frame_shape):
        filtered_rects, centers = self.rectangle_filter(contours)
        labels = self.classify_points(centers)
        rect_new = self.merge_rectangles(labels, filtered_rects, frame_shape)
        return rect_new

    def find_intersection(self, m_list):
        for i, v in enumerate(m_list):
            for j, k in enumerate(m_list[i + 1:], i + 1):
                if np.in1d(v, k).any():
                    m_list[i] = np.union1d(v, m_list.pop(j)).tolist()
                    return self.find_intersection(m_list)
        return m_list

    def classify_points(self, points):
        # 计算距离矩阵
        distance_list = np.zeros((len(points), len(points)))
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance_list[i][j] = ((points[i][0] - points[j][0]) ** 2 + (
                        points[i][1] - points[j][1]) ** 2) ** 0.5  # 计算两点之间的距离

        distance_all = distance_list + distance_list.T - np.diag(np.diag(distance_list))
        # 中心点分类
        temp = []
        for i in range(len(points)):
            indices = np.where(distance_all[i] <= 180)[0].tolist()
            temp.append(indices)
        classes = self.find_intersection(temp)
        return classes

    def merge_rectangles(self, classes, rects, frame_shape):
        rects_new = []
        for cla in classes:
            min_x = np.min(rects[cla, 0])
            min_y = np.min(rects[cla, 1])
            max_x = np.max(rects[cla, 2])
            max_y = np.max(rects[cla, 3])
            [x1, y1, x2, y2] = [min_x, min_y, max_x, max_y]
            # 判断矩形框是否满足要求
            if (x2 - x1) % 32 != 0:
                x2 = x2 + (32 - (x2 - x1) % 32)  # 扩大宽度
                if x2 > frame_shape[1]:
                    x2 = frame_shape[1]
                    if (x2 - x1) % 32 != 0:
                        x2 = x2 - ((x2 - x1) % 32)  # 缩减宽度
            if (y2 - y1) % 32 != 0:
                y2 = y2 + (32 - (y2 - y1) % 32)  # 扩大宽度
                if y2 > frame_shape[0]:
                    y2 = frame_shape[0]
                    if (y2 - y1) % 32 != 0:
                        y2 = y2 - ((y2 - y1) % 32)  # 缩减宽度
            if (x2 - x1) <= 32 or (y2 - y1) <= 32:
                continue
            rects_new.append([x1, y1, x2, y2])
        return rects_new

    def run(self, frame):
        results = {}

        fg_mask = self.bg_subtractor.apply(frame)

        opening = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        dilated = cv2.dilate(opening, self.kernel, iterations=1)

        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rect_new = self.rect_fuse(contours, frame.shape)

        if contours == []:
            return results
        # test = frame.copy()

        res_cla = []
        res_score = []
        res_rect = []
        # 遍历每个轮廓，绘制矩形框并裁剪出区域·
        for rect in rect_new:
            [x1, y1, x2, y2] = rect
            # cv2.rectangle(test, (x1, y1), (x2, y2), (0, 0, 255), 2)
            roi = frame[y1:y2, x1:x2]

            pre_results = self.model.predictor(roi)

            result_rect = pre_results[0].boxes.xyxy.numpy()
            result_rect[:, [0, 2]] += x1
            result_rect[:, [1, 3]] += y1
            result_rect = [[int(x) for x in i] for i in result_rect.tolist()]
            res_cla.extend([int(x) for x in pre_results[0].boxes.cls.numpy().tolist()])
            res_score.extend([round(x, 3) for x in pre_results[0].boxes.conf.numpy().tolist()])
            res_rect.extend(result_rect)

        results['result_category'] = res_cla
        results['result_score'] = res_score
        results['result_rect'] = res_rect

        return results


def draw(frame, results):
    for i in range(len(results['result_rect'])):
        cv2.rectangle(frame, (int(results['result_rect'][i][0]), int(results['result_rect'][i][1])),
                      (int(results['result_rect'][i][2]), int(results['result_rect'][i][3])),
                      (0, 255, 0), 2)
        cv2.putText(frame, 'score: ' + str(results['result_score'][i]),
                    (int(results['result_rect'][i][0]), int(results['result_rect'][i][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'class: ' + str(results['result_category'][i]),
                    (int(results['result_rect'][i][0]), int(results['result_rect'][i][1]) - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
    return frame

if __name__ == "__main__":
    time1 = time.time()
    # video = cv2.VideoWriter('result_cut.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (960, 540))
    image_preprocess = image_preprocess()
    video_path = r'E:\DemoProject\5min_test_video.mp4'
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # video_capture.set(cv2.CAP_PROP_POS_FRAMES, 3430)
    temp = 0
    while (1):
        ret, frame = video_capture.read()
        if not ret:
            break
        if temp % 5 == 0:
            print(total_frames, '-->', temp)
            frame = cv2.resize(frame, (960, 540))

            frame = image_preprocess.distortion_correction(frame)
            results = image_preprocess.run(frame)

            frame = draw(frame, results)

            # video.write(frame)
        temp += 1
    time2 = time.time()
    print('总耗时：', time2 - time1)

    video_capture.release()
    # video.release()

    # 总耗时： 167.57258772850037