import os
import cv2
import math
import pickle
import numpy as np
from time import time
import face_recognition
from sklearn import neighbors
from translate import Translator
from arabic_content import ARABIC_MAPPING
from sklearn.model_selection import train_test_split
from face_recognition.face_recognition_cli import image_files_in_folder


class MafQudRecognition:
    def __init__(self):
        self.people = []
        self.features = []
        self.ids = []
        self.face_locations = []
        self.knn_clf = None

    def import_data(self, ids_DIR='ids.npy', people_DIR='people.npy', fetaures_DIR='feature.npy', face_locations_DIR='face_location.npy'):
        """
        Import the .npy saved data to Mafqud_Recongition instance. 

        Parameters
        ----------
        ids_DIR : str, optional
            directory of ids.npy. The default is 'ids.npy'.
        people_DIR : str, optional
            directory of people.npy. The default is 'people.npy'.
        fetaures_DIR : str, optional
            directory of feature.npy. The default is 'feature.npy'.

        Returns
        -------
        None.

        """
        print("Importing the data ...")
        self.ids = np.load(ids_DIR, allow_pickle=True)
        self.ids = self.ids.astype('str')
        self.people = np.load(people_DIR, allow_pickle=True)
        self.people = self.people.astype('str')
        self.features = np.load(fetaures_DIR, allow_pickle=True)
        self.features = self.features.astype('float')
        self.face_locations = np.load(face_locations_DIR, allow_pickle=True)
        self.face_locations = list(self.face_locations)
        print("Data imported successfully!")
        print("="*70)
        print("Data Summary: ")
        print("Number of people: {}".format(len(self.people)))
        print("Number of features: {}".format(len(self.features)))
        print("Number of ids: {}".format(len(self.ids)))
        print("Number of face_locations: {}".format(len(self.face_locations)))

    def append_data(self, imgDir, name):
        """
        Append new data to the system data.

        Parameters
        ----------
        imgDir : str
            the directory of the images data.
        name : str
            name of the person to be appended.

        Returns
        -------
        None.

        """
        face_location, face_encoding = self.detect_face_location(imgDir)
        if face_location is not None: 
            self.import_data()   ## ToDo in future (if the data not found) 
            if list(self.people).count(name) > 0: 
                id = list(self.people).index(name)
                self.ids = np.append(self.ids, id)
            else:
                self.people = np.append(self.people, name)
                id = len(self.people)-1
                self.ids = np.append(self.ids, id)
            self.features = np.vstack([self.features, np.array(face_encoding)])
            print("Person: {} is appended in system data with id: {}".format(name, id))
            print("="*70)
            print("Data Summary: ")
            print("Number of people: {}".format(len(self.people)))
            print("Number of features: {}".format(len(self.features)))
            print("Number of ids: {}".format(len(self.ids)))
            print("Number of face_locations: {}".format(len(self.face_locations)))
            print("="*70)
            self.training_classifier(train_test=False)
        
    
    def detect_face_location(self, image_path, searching_model="hog"):
        """
        Detect the face location and return the location and encoding. 

        Parameters
        ----------
        image_path : str
             The directory of the images data.
         searching_model : str, optional 
            The method will be used to search for the location of the face. default is "hog". 
            Methods: 
                - **hog**: for faster search, but less accurate results. 
                - **cnn**: for more accurate results, but slower search. 
                - **mix**: will use hog first, if none then will use cnn (preferable). 
        Returns
        -------
        face_coordinates : tuple
            the coordinates of the location of face, if not found return None.
        face_encoding : list
            the encodings of the face, if not found return None.

        """
        image = face_recognition.load_image_file(image_path) 
        searching_model = searching_model.lower().strip()
        if searching_model == "mix":
            print("Try first with hog ...")
            face_coordinates = face_recognition.face_locations(image, model="hog")
            if len(face_coordinates) >= 1: 
                face_encoding = face_recognition.face_encodings(image, known_face_locations=face_coordinates)
                return face_coordinates, face_encoding
            else:
                print("Not detected with hog, try again with cnn ...")
                face_coordinates = face_recognition.face_locations(image, model="cnn")
                if len(face_coordinates) >= 1: 
                    face_encoding = face_recognition.face_encodings(image, known_face_locations=face_coordinates)
                    return face_coordinates, face_encoding
                else: 
                    print("No face detected at all with {} (hog & cnn) models".format(searching_model))
                    return None, None
        else: 
            face_coordinates = face_recognition.face_locations(image, model=searching_model)
            if len(face_coordinates) >= 1: 
                face_encoding = face_recognition.face_encodings(image, known_face_locations=face_coordinates) ###
                return face_coordinates, face_encoding
            else: 
                print("No face detected with ({}) model".format(searching_model))
                return None, None
                    
            
    def create_data(self, images_DIR, face_loc_model="hog"):
        """
        Find face coordinates and extract its encodings.
        Save encodings in feature.npy, ids in ids.npy and names in people.npy in same directory.

        Parameters
        ----------
        images_DIR : str
            The directory of the images data.
        face_loc_model : str, optional 
            The method will be used to search for the location of the face. default is "hog". 
            Methods: 
                - **hog**: for faster search, but less accurate results. 
                - **cnn**: for more accurate results, but slower search. 
                - **mix**: will use hog first, if none then will use cnn (preferable). 

        Returns
        -------
        ids : List
            List of indices in people list
        people : List
            List of names of people
        features : List
            List of face encodings of each person
        face_locations : List
            List of face location
        """
        for person in os.listdir(images_DIR):
            self.people.append(person)
        t0 = time()
        print("Loading Dataset \n\n")
        for id, person in enumerate(self.people):
            print("="*70)
            print(f"Loading Person: {person}")
            for img in image_files_in_folder(os.path.join(images_DIR, person)):
                print(f"Loading Image: {img}")
                face_coordinates, face_encoding = self.detect_face_location(img, searching_model=face_loc_model)
                if face_coordinates is not None:
                    self.face_locations.append(face_coordinates)
                    self.features.append(face_encoding[0])
                    self.ids.append(id)
        print("\n")            
        np.save('face_locations.npy', np.array(self.face_locations, dtype=object))
        np.save('feature.npy', np.array(self.features, dtype=float))
        np.save('ids.npy', np.array(self.ids, dtype=str))
        np.save('people.npy', np.array(self.people, dtype=str))
        print("="*70)
        print("Data Summary: ")
        print("Number of people: {}".format(len(self.people)))
        print("Number of features: {}".format(len(self.features)))
        print("Number of ids: {}".format(len(self.ids)))
        print("Number of face_locations: {}".format(len(self.face_locations)))
        print("="*70)
        print("Data is successfully created and loaded in : {:.2f}s".format(time()-t0))
        return self.ids, self.people, self.features, self.face_locations


    def training_classifier(self, model_save_path=None, train_test=True, n_neighbors=None, knn_algo='ball_tree', verbose=False):
        """
        Compare and classify encodings (with possibility to save it).

        Parameters
        ----------
        model_save_path : str, optional
            The path where the model will be saved in. default is None. 
        train_test : bool, optional
            Indicate wether you will train in all the data or you will make train test split for the data. default is True.
        Returns
        -------
        knn_clf: model
            The model that classifies images using KNN algorithm.
        """
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(self.features))))
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)
        if train_test == True:
            print("Trained in %70 of the data, and tested in %30, expected true accuracy")
            X_train, X_test, y_train, y_test = train_test_split(self.features, self.ids, test_size=0.3,train_size=.7, random_state=42)
        else: 
            print("Trained in all data, expected %100 accuracy (not true)")
            X_train, y_train = self.features, self.ids
            X_test, y_test = self.features, self.ids
        print("Start trainging -------------------")
        t0 = time()
        self.knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        self.knn_clf.fit(X_train, y_train)
        print("Accuracy:   %{:.2f}".format(self.knn_clf.score(X_test, y_test)*100))
        print("Successfully trained  in : {:.2f}s".format(time() - t0))
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(self.knn_clf, f)

        return self.knn_clf

    def predict(self, unkown_img_path, model_path=None, face_loc_model="hog", distance_threshold=0.6):
        """
        Recognize the person in picture.

        Parameters
        ----------
        unkown_img_path : str
            Path to image
        face_loc_model : str, optional 
            The method will be used to search for the location of the face. default is "hog". 
            Methods: 
                - **hog**: for faster search, but less accurate results. 
                - **cnn**: for more accurate results, but slower search. 
                - **mix**: will use hog first, if none then will use cnn (preferable). 
        model_path : str, optional
            The path of the saved model (.clf file).
        distance_threshold: float
            Determine the distance
        Returns
        -------
        (pred, loc): tuple
            the location of the face and predicted id
        """
        print("Searching For Matching Person ...")
        print("Please Wait ...")
        t0 = time()
        if self.knn_clf is None:
            with open(model_path, 'rb') as f:
                self.knn_clf = pickle.load(f)
                
        face_location, face_encoding = self.detect_face_location(unkown_img_path, searching_model=face_loc_model)
        if face_location is not None:
            closest_distances = self.knn_clf.kneighbors(face_encoding, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_location))]
    
            matching = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                zip(self.knn_clf.predict(face_encoding), face_location, are_matches)]
            
            print("="*70)
            if matching[0][0] == "unknown": 
                print("Unknown person")
            else:
                print("Finshed searching in {:.2f}s".format(time()-t0))
                print("Found person with id: {}".format(matching[0][0]))
                print("Name: {}".format(self.people[int(matching[0][0])]))
                
            return matching
        else: 
            return -1
        
        
          
    def mapping_to_english(self, name):
        """
         Mapping the Arabic name to English using list of 
         pre-written dict according to ARABIC_MAPPING.
         
        Parameters
        ----------
        name : str
            list of Arabic names to be mapped.
    
        Returns
        -------
        names_mapped : str
            string of the mapped name.
    
        """
        mapped_name = ""
        for c in name: 
            try: 
                mapped_name += ARABIC_MAPPING[c]
            except: 
                mapped_name += c
        
        return mapped_name.title()


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


    def draw_box(self, image, face_location=None, nameId=None, unknown=False, mapping_method="translating", frame_thickness=3, font_thickness=2):
        """
        Draw a box (rectangle) around the image, and put the name of the predicted person.
        ----------
        image : list
            the image of the person.
        face_location : list, optional.
            the coordinates of the face of the person. default is None
        nameId : string, optional
            the id of the person in ids. default is None
        unknown : bool, optional 
            indicating if the person in the photo is unknown. default is False. 
        mapping_method : str, optional 
            how to map the arabic name of the person if it is found. default is "translating"
            Options: 
                - **translating**: online translating the name using translate open source api. 
                - **mapping**: offline mapping the Arabic name to English (using ARABIC_MAPPING dict).
                - **leave**: leave the Arabic name as it is. 
        frame_thickness : int, optional
            the thickness of the frame with which the frame plotted. default is 3
        font_thickness : int, optional
            thickness of the font with which the name typed. default is 2
        Returns
        -------
        (pred, loc): tuple
            the location of the face and predicted id
        """
        if not unknown:
            name = self.people[int(nameId)]
            name = " ".join(name.split()[:2])
            mapping_method = mapping_method.lower().strip()
            if mapping_method == "translating": 
                name = self.translate_content(name)
            elif mapping_method == "mapping": 
                name = self.mapping_to_english(name)
            else:
                pass
        else:
            name = "unknown"
        
        img_array = face_recognition.load_image_file(image)
        image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        if face_location is not None: 
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

        cv2.imshow("MafQud Recognition", image)
        cv2.waitKey(5000)
        cv2.destroyWindow("MafQud Recognition")
