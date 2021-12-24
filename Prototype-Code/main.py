from MafQudRecognition import MafQudRecognition
imgDir = r'C:\Users\omar_\PycharmProjects\FaceReognistion\202109165614657462504050423.jpg'
DIR = r'C:\Users\omar_\PycharmProjects\FaceReognistion\img'
X = MafQudRecognition()

# ids , people, features, face_location = X.create_encodings(DIR, face_loc_model='hog')
# print(f"ids {len(ids)}")
# print(f"people {len(people)}")
# print(f"features {len(features)}")
# print(f"face_location {len(face_location)}")
# X.training_classifier("knn_model.clf", n_neighbors=2)
x = X.predict(unkown_img_path=imgDir,model_path='knn_model.clf')
if x[0][0] == "unknown":
    X.draw_box(imgDir, x[0][1], x[0][0], True)
    input_id = input("Enter ID(BackEnd): ")
    input_Name = input("Enter Name(BackEnd): ")
    X.append_encoding(imgDir,input_id, input_Name)
else:
    X.draw_box(imgDir, x[0][1], x[0][0], False)
