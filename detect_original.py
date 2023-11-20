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

    def distortion_correction(self, image):
        image = cv2.resize(image, (960, 540))
        mat_intri = np.array([[603.95326447, 0, 481.36708033],
                              [0, 604.51242483, 272.66842792],
                              [0, 0, 1]])
        coff_dis = np.array([-3.89127437e-01, 1.92954238e-01, -3.28167575e-04, -2.74500971e-04, -5.41705555e-02])
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_intri, coff_dis, (960, 540), 0, (960, 540))
        dis_image = cv2.undistort(image, mat_intri, coff_dis, None, newcameramtx)
        return dis_image



    def run(self, frame):
        results = {}

        pre_results = self.model.predictor(frame)

        results['result_category'] = [int(x) for x in pre_results[0].boxes.cls.numpy().tolist()]
        results['result_score'] = [round(x, 3) for x in pre_results[0].boxes.conf.numpy().tolist()]
        results['result_rect'] = [[int(x) for x in i] for i in pre_results[0].boxes.xyxy.numpy().tolist()]

        return results

def draw(frame, results):
    for i in range(len(results['result_rect'])):
        frame = cv2.rectangle(frame, (int(results['result_rect'][i][0]), int(results['result_rect'][i][1])), (int(results['result_rect'][i][2]), int(results['result_rect'][i][3])),
                                    (0, 255, 0), 2)
        frame = cv2.putText(frame, 'score: ' + str(results['result_score'][i]), (int(results['result_rect'][i][0]), int(results['result_rect'][i][1])),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        frame = cv2.putText(frame, 'class: ' + str(results['result_category'][i]),
                            (int(results['result_rect'][i][0]), int(results['result_rect'][i][1])-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
    return frame


if __name__ == "__main__":
    time1 = time.time()
    # video = cv2.VideoWriter('result_original.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (960, 540))
    image_preprocess = image_preprocess()
    video_path = r'E:\DemoProject\5min_test_video.mp4'
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # video_capture.set(cv2.CAP_PROP_POS_FRAMES, 1375)
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
    print('总耗时：', time2-time1)

    video_capture.release()
    # video.release()

    # 总耗时： 366.95037722587585
