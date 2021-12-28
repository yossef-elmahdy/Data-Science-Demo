import os
from Mafqud_Recognition import MafQudRecognition 



mafqud = MafQudRecognition()
images_DIR = r"** Directory of images to create encodings **"  

'''
# For creating the data and training the model
try: 
    mafqud.create_data(images_DIR, face_loc_model="hog")
    mafqud.training_classifier(train_test=False)   
except:
    print("Please FILL in in the variable images_DIR with required directory of images")
# mafqud.import_data()
'''

# For predicting the data (testing)
unknwon_image_DIR = r"** Directory of testing images (query images) **"
try: 
    for img in os.listdir(unknwon_image_DIR): 
        path = os.path.join(unknwon_image_DIR, img) 
        pred = mafqud.predict(path, face_loc_model="hog" , model_path="knn_clf.clf" )
        if pred == -1:
            print("No face in the photo")
            mafqud.draw_box(path, unknown=True)
        elif pred[0][0] == "unknown":
            mafqud.draw_box(path, face_location=pred[0][1], unknown=True)
            print("We will add this person to our model ...")
            name = input("Enter the name in Arabic: ")
            mafqud.append_data(path, name)
        else:
            mafqud.draw_box(path, face_location=pred[0][1], nameId=pred[0][0], mapping_method="mapping")
except:
    print("Please FILL in in the variable unknown_image_DIR with required directory of images")
    
 