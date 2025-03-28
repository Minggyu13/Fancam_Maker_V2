import cv2
import supervision as sv
from ultralytics import YOLO
import matplotlib.pyplot as plt

class FaceDetection:
    def __init__(self, detection_model_path):
        self.model = YOLO(detection_model_path)


    def detect_face(self, image, visualize=False):

        results = self.model(image, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results).with_nms()

        if visualize:
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            annotated_image = image.copy()
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

            sv.plot_image(annotated_image)

        return detections.xyxy

    @staticmethod
    def extract_face(face_boxes,image, visualize=False):
        resized_face_arr = []
        face_box_arr = []

        if len(face_boxes) == 0:

            return None, None

        else:

            for face_box in face_boxes:
                x1, y1, x2, y2 = map(int, face_box)  # 좌표 정수 변환
                w, h = x2 - x1, y2 - y1  # 너비, 높이 계산

                face = image[y1:y1 + h, x1:x1 + w]

                resized_face = cv2.resize(face, (160, 160))


                face_box_arr.append((x1, y1, w, h))
                resized_face_arr.append(resized_face)


            if visualize:
                for resized_face in resized_face_arr:
                    plt.imshow(resized_face)
                    plt.axis("off")  # 축 숨기기 (선택 사항)
                    plt.show()


            return resized_face_arr, face_box_arr