import cv2
import numpy as np
from ml_service.isedol_face_recognition import FaceRecognition
from ml_service.isedol_face_detection import FaceDetection

class Tracking:
    def __init__(self, tracker, video_path):
        self.output_size = (375, 667)
        self.fit_to = 'height'
        self.tracker = tracker
        self.cap = cv2.VideoCapture(video_path)
        self.fourcc =cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), self.fourcc, self.cap.get(cv2.CAP_PROP_FPS),self.output_size)
        self.face_detector = FaceDetection(detection_model_path="/Users/mingyu/Desktop/Fancam_Maker_V2/saved_models/isedol_face_detector2.pt")
        self.face_recognizer = FaceRecognition(labeled_face_embeddings_path='/Users/mingyu/Desktop/Fancam_Maker_V2/saved_models/isedol_face_embedding_dataset2.npz',trained_svm_classifier_path='/Users/mingyu/Desktop/Fancam_Maker_V2/saved_models/isedol_fancam_svm_model.pkl')



    def tracking_idol(self, idol_name):
        if not self.cap.isOpened():
            exit()

        top_bottom_list, left_right_list = [], []
        predict_label = str()
        count = 0
        init_count = 0
        rect = list()



        while True:
            ret, img = self.cap.read()
            face_detection_boxes = self.face_detector.detect_face(img, visualize=True)
            extracted_faces_arr, extracted_faces_box_arr = self.face_detector.extract_face(face_detection_boxes, img,visualize=True)
            init_count += 1
            if extracted_faces_arr is not None:
                for idx, extracted_face in enumerate(extracted_faces_arr):
                    extracted_face_embedding = self.face_recognizer.get_embedding(extracted_face)
                    predict_label = self.face_recognizer.predict(extracted_face_embedding)

                    if predict_label == idol_name:
                        rect = extracted_faces_box_arr[idx]
                        print(rect)
                        break

            else:
                continue

            if predict_label == idol_name:
                break

        print(init_count)
        print(rect)


        while True:
            count += 1
            # read frame from video
            ret, img = self.cap.read()

            if not ret:
                exit()
            # update tracker and get position from new frame
            success, box = self.tracker.update(img)
            # if success:
            left, top, w, h = [int(v) for v in box]
            right = left + w
            bottom = top + h

            # save sizes of image
            top_bottom_list.append(np.array([top, bottom]))
            left_right_list.append(np.array([left, right]))

            # use recent 10 elements for crop (window_size=10)
            if len(top_bottom_list) > 10:
                del top_bottom_list[0]
                del left_right_list[0]

            # compute moving average
            avg_height_range = np.mean(top_bottom_list, axis=0).astype(int)
            avg_width_range = np.mean(left_right_list, axis=0).astype(int)
            avg_center = np.array([np.mean(avg_width_range), np.mean(avg_height_range)])  # (x, y)

            # compute scaled width and height
            scale = 1.3
            avg_height = (avg_height_range[1] - avg_height_range[0]) * scale
            avg_width = (avg_width_range[1] - avg_width_range[0]) * scale

            # compute new scaled ROI
            avg_height_range = np.array([avg_center[1] - avg_height / 2, avg_center[1] + avg_height / 2])
            avg_width_range = np.array([avg_center[0] - avg_width / 2, avg_center[0] + avg_width / 2])

            # fit to output aspect ratio
            if self.fit_to == 'width':
                avg_height_range = np.array([
                    avg_center[1] - avg_width * self.output_size[1] / self.output_size[0] / 2,
                    avg_center[1] + avg_width * self.output_size[1] / self.output_size[0] / 2
                ]).astype(int).clip(0, 9999)

                avg_width_range = avg_width_range.astype(int).clip(0, 9999)
            elif self.fit_to == 'height':
                avg_height_range = avg_height_range.astype(int).clip(0, 9999)

                avg_width_range = np.array([
                    avg_center[0] - avg_height * self.output_size[0] / self.output_size[1] / 2,
                    avg_center[0] + avg_height * self.output_size[0] / self.output_size[1] / 2
                ]).astype(int).clip(0, 9999)

            # crop image
            result_img = img[avg_height_range[0]:avg_height_range[1], avg_width_range[0]:avg_width_range[1]].copy()

            # resize image to output size
            result_img = cv2.resize(result_img, self.output_size)

            # visualize
            pt1 = (int(left), int(top))
            pt2 = (int(right), int(bottom))
            cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)

            cv2.imshow('img', img)
            cv2.imshow('result', result_img)
            # write video
            self.out.write(result_img)
            if cv2.waitKey(1) == ord('q'):
                break

        # release everything
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


    def compute_average_move(self):
        pass


video_path = "/Users/mingyu/Desktop/Fancam_Maker_V2/video_files/[MV] 달이 아름다워｜Cover by 주르르.mp4"
cap = cv2.VideoCapture(video_path)
tracking = Tracking(tracker=cv2.TrackerMIL_create(), video_path=video_path)
tracking.tracking_idol(idol_name='jururu')