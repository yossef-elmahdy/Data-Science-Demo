import os
from Mafqud_Recognition import MafQudRecognition 



mafqud = MafQudRecognition()
images_DIR = r"Image DIR"


# For creating the data and training the model
# try:
#     mafqud.create_data(images_DIR, face_loc_model="hog")
#     mafqud.training_classifier(train_test=False,model_save_path='knn_model.clf')
# except FileNotFoundError:
#     print("Directory not found ...")
#     print("Please FILL in in the variable images_DIR with required directory of images")


# For predicting the data (testing)
# unknwon_images_DIR = r"Image DIR"
# mafqud.import_data()
# try: 
#     for img in os.listdir(unknwon_images_DIR):
#         path = os.path.join(unknwon_images_DIR, img)
#         pred = mafqud.predict(path, face_loc_model="hog" , model_path="knn_model.clf")
#         if pred == -1:
#             print("No face in the photo")
#         elif pred[0][0] == "unknown":
#             mafqud.draw_box(path, face_location=pred[0][1], unknown=True)
#             print("Unknown Face")
#         else:
#             mafqud.draw_box(path, face_location=pred[0][1], nameId=pred[0][0], mapping_method="mapping")
# except FileNotFoundError: 
#     print("Directory not found ...")
#     print("Please FILL in in the variable unknown_images_DIR with required directory of images")


# For appending new person to the model
# images_user_add = r"Image DIR"
# mafqud.import_data()
# print("We will add this person to our model ...")
# name = input("Enter the name in Arabic: ")
# try: 
#     mafqud.append_data(images_user_add, name)
# except FileNotFoundError: 
#     print("Directory not found ...")
#     print("Please FILL in in the variable images_user_add with required directory of images")







