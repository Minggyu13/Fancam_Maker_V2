import pickle
from keras_facenet import FaceNet
import numpy as np
from sklearn.preprocessing import LabelEncoder


class FaceRecognition:
    def __init__(self,labeled_face_embeddings_path, trained_svm_classifier_path):
        self.labeled_face_embeddings = np.load(labeled_face_embeddings_path)
        self.Y = self.labeled_face_embeddings['arr_1']
        self.encoder = LabelEncoder().fit(self.Y)
        self.model = pickle.load(open(trained_svm_classifier_path, 'rb'))
        self.embedder = FaceNet()


    def get_embedding(self, face_img):
        face_image = face_img.astype('float32') # 3D (160 x 160 x 3)
        face_image = np.expand_dims(face_image, axis=0) # 4D (none x 160 x 160 x 3)
        y_hat = self.embedder.embeddings(face_image)

        return y_hat[0] # 512D (1 x 1 x 512)


    def predict(self, embedded_face_img):
        pred = self.model.predict(np.array([embedded_face_img]))
        predicted_label = self.encoder.inverse_transform(pred)

        return predicted_label
