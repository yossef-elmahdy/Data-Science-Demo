import os
import cv2
import math
import pickle
import numpy as np
from time import time
import face_recognition
from sklearn import neighbors
from translate import Translator
from sklearn.model_selection import train_test_split
from face_recognition.face_recognition_cli import image_files_in_folder


class MafQudRecognition:
    def __init__(self):
        self.people = []
        self.features = []
        self.ids = []
        self.face_location = []
        self.knn_clf = None

    def import_encodings(self):
        self.ids = np.load('ids.npy', allow_pickle=True)
        self.ids = self.ids.astype('str')
        self.people = np.load('people.npy', allow_pickle=True)
        self.people = self.people.astype('str')
        self.features = np.load('feature.npy', allow_pickle=True)
        self.features= self.features.astype('float')

    def append_encoding(self, imgDir, id, name):
        img = face_recognition.load_image_file(imgDir)
        face_locations = face_recognition.face_locations(img)
        if len(face_locations) >= 1:
            faces_encodings = face_recognition.face_encodings(img, known_face_locations=face_locations)
        else:
            print("No face detected")
            return

        print(type(faces_encodings))
        self.features = np.load('feature.npy', allow_pickle=True)
        self.features = self.features.astype('float')
        print(len(self.features))
        self.features = np.vstack([self.features,np.array(faces_encodings)])
        print(len(self.features))
        self.ids = np.load('ids.npy', allow_pickle=True)
        self.ids = self.ids.astype('str')
        print(len(self.ids))
        self.ids = np.append(self.ids, id)
        print(len(self.ids))
        self.people = np.load('people.npy', allow_pickle=True)
        self.people = self.people.astype('str')
        self.people = np.append(self.people, name)
        #self.training_classifier("knn_model.clf", n_neighbors=2)

    def create_encodings(self, DIR, face_loc_model="cnn"):
        """
        Find face coordinates and extract its encodings.
        Save encodings in feature.npy, ids in ids.npy and names in people.npy in same directory.

        Parameters
        ----------
        DIR : str
            The path  of the training data

        Returns
        -------
        ids : List
            List of indices in people list
        people : List
            List of names of people
        features : List
            List of face encodings of each person
        face_location: List
            List of face location
        """


        for path in os.listdir(DIR):
            self.people.append(path)
        t0 = time()
        print("Loading Dataset \n\n")
        for id, person in enumerate(self.people):
            print(f"Loading Person: {person} \n")
            for img in image_files_in_folder(os.path.join(DIR, person)):
                print(f"Loading Image: {img}")

                image = face_recognition.load_image_file(img)
                face_bounding_boxes = face_recognition.face_locations(image, model=face_loc_model)
                if len(face_bounding_boxes) >= 1:
                    encoding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
                    self.face_location.append(face_bounding_boxes)
                    self.features.append(encoding)
                    self.ids.append(id)
                else:
                    print("No face detected")
        np.save('face_location.npy', np.array(self.face_location, dtype=object))
        np.save('feature.npy', np.array(self.features, dtype=float))
        np.save('ids.npy', np.array(self.ids, dtype=str))
        np.save('people.npy', np.array(self.people, dtype=str))
        t1 = time() - t0
        print(f"Successfully trained  in : {time() - t0}s")
        return self.ids, self.people, self.features, self.face_location


    def training_classifier(self, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
        """
        Compare and classify encodings.

        Parameters
        ----------
        ids : List
            List of indices in people list
        features : List
            List of face encodings of each person
        Returns
        -------
        knn_clf: model
            Classifies images
        """
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(self.features))))
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.ids, test_size=0.3,train_size=.7, random_state=42)
        print("Start trainging -------------------")
        t0 = time()
        self.knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        self.knn_clf.fit(X_train, y_train)
        print(f"Accuracy:   {self.knn_clf.score(X_test, y_test)}")
        print(f"Successfully trained  in : {time() - t0}s")
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(self.knn_clf, f)

        return self.knn_clf

    def predict(self, unkown_img_path, model_path=None, distance_threshold=0.6):
        """
        Recognize the person in picture.

        Parameters
        ----------
        unkown_img_path : string
            Path to image
        knn_clf : model
            To directly specify model
        distance_threshold: float
            Determine the distance
        Returns
        -------
        (pred, loc): tuple
            the location of the face and predicted id
        """
        if self.knn_clf is None:
            with open(model_path, 'rb') as f:
                self.knn_clf = pickle.load(f)

        unkown_img = face_recognition.load_image_file(unkown_img_path)
        unkown_face_locations = face_recognition.face_locations(unkown_img)
        if len(unkown_face_locations) >= 1:
            faces_encodings = face_recognition.face_encodings(unkown_img, known_face_locations=unkown_face_locations)
        else:
            print("No face detected")
            return
        closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(unkown_face_locations))]

        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                zip(self.knn_clf.predict(faces_encodings), unkown_face_locations, are_matches)]

    def translate_content(self, name_english, from_language='ar', to_language='en'):
        """
        Translate the content (mostly: names, govs) list from one
        language (mostly: 'ar') to another (mostly: 'en') using
        translate open source library.
        Note: has daily limited times of usage.

        Parameters
        ----------
        contents : list
            the content list to be translated (mostly: names, govs).
        from_language : str, optional
            the language of your content. The default is 'ar'.
        to_language : str, optional
            the language you need to translate to. The default is 'en'.

        Returns
        -------
        contents_translated : list
            list of the translated content.

        """
        translator = Translator(from_lang=from_language, to_lang=to_language)

        name_translated = translator.translate(name_english)

        return name_translated

    def draw_box(self, image, face_location, nameId, unkown=False, frame_thickness=3, font_thickness=2):
        """
        Draw a box (rectangle) around the image, and put the name of the predicted person.
        ----------
        image : list
            the image of the person.
        face_location : list
            the coordinates of the face of the person.
        name : string
            the name of the person.
        frame_thickness : int
            the thickness of the frame with which the frame plotted.
        font_thickness : int
            thickness of the font with which the name typed.
        Returns
        -------
        (pred, loc): tuple
            the location of the face and predicted id
        """
        if not unkown:
            name = self.people[int(nameId)]
        else:
            name = "unkown"
        img_array = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        #name = " ".join(self.translate_content(name).split(" ")[:2])
        # #Draw Rectangle
        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])
        color = [0, 156, 0]
        cv2.rectangle(image, top_left, bottom_right, color, frame_thickness)
        #
        # Draw Text
        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2] + 22)

        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, name, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), font_thickness)

        cv2.imshow(name, image)
        cv2.waitKey(10000)
        cv2.destroyWindow("window")
