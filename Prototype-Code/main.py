import os
from Mafqud_Recognition import MafQudRecognition 



mafqud = MafQudRecognition()
images_DIR = r"C:\Users\yosse\Desktop\Graduation Project\Implementation\Data-Science\Scrapping Codes\Scrapped_Data"

# mafqud.create_data(images_DIR, face_loc_model="mix")
#mafqud.import_data()
#mafqud.training_classifier(train_test=False)   
unknwon_image_DIR = r"C:\Users\yosse\Desktop\Graduation Project\Implementation\Data-Science\Scrapping Codes\Test"
for img in os.listdir(unknwon_image_DIR): 
    path = os.path.join(unknwon_image_DIR, img) 
    pred = mafqud.predict(path, face_loc_model="hog", model_path="knn_clf.clf")
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
    
 