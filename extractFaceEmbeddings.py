#4 extracting embeddings/ navigating through files
import os
import pickle

#general structures
from collections import defaultdict
from numpy import array

#Image manipuation packages
import face_recognition


image_dataset = "Dataset"
outFolder = "Serialized/embedded.p"

"""Returns the name of people in the dataset, 
organizes the images into a dictionary for quickly going through files,
return the total amount of images in the dataset"""
def filePaths(rootFolder:str) -> tuple:
    people = os.listdir(image_dataset)
    images = defaultdict(list)
    totalImages = 0

    for subdir, dirs, files in os.walk(rootFolder):
        if subdir != rootFolder:
            images[subdir[len(rootFolder)+1:]] = files
            totalImages += len(files)
    return (people, images, totalImages)

def embeddings(files)-> defaultdict:
    embbed = defaultdict(array)

    for person in files:
        initEmb = []
        for image in files[person]:
            path = image_dataset +"/" +person + "/" + image
            loaded_image = face_recognition.load_image_file(path , mode="RGB")
            faces = face_recognition.face_locations(loaded_image)
            if 0< len(faces) <2:
                currentEncode = face_recognition.face_encodings(loaded_image)
                initEmb.append(currentEncode[0].tolist())

        embbed[person] = initEmb
    return embbed



people, files, numImages = filePaths(image_dataset)
embedded =  embeddings(files)
pickle.dump(embedded, open(outFolder, "wb"))







#cap = cv2.VideoCapture(0)
#while True:
#    sucess, img = cap.read()
#    #imgS = cv2.cvtColor(cv2.resize(img, (0,0),None, 1,1), cv2.COLOR_BGR2RGB)
#    #imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#    location = face_recognition.face_locations(img)
#    if len(location)>0:
#        #encode = face_recognition.face_encodings(img)[0]
#        pass
#    cv2.imshow("Application", img)
#    key = cv2.waitKey(1)
#    if key== ord('q'):
#        break
#cv2.destroyAllWindows()
