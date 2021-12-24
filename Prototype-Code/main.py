from Mafqud_Recognition import MafQudRecognition 



mafqud = MafQudRecognition()
images_DIR = r"C:\Users\yosse\Desktop\Graduation Project\Implementation\Data-Science\Scrapping Codes\Scrapped_Data"


mafqud.create_data(images_DIR, face_loc_model="hog")