import pickle
import face_recognition
from collections import defaultdict
import cv2
from numpy import average

embeddingsFile = "Serialized/embedded.p"

def recoverEmbeddings(pickleFile:str):
    return pickle.load(open(pickleFile, "rb"))

def checkImage(imgPath:str, embeddings):
    distances = defaultdict(list)
    testImage = face_recognition.load_image_file(imgPath, mode="RGB")
    encode = face_recognition.face_encodings(testImage)
    if len(encode)>0:
        facialFeatures = encode[0]
        for person in embeddings:
            distance = face_recognition.face_distance(embeddings[person], facialFeatures)
            distances[person] = distance
    return distances

def figureOut(distances, image):
    minimum = 1
    label = "Unknown"
    for person in distances:
        avg = average(distances[person])
        if avg < 0.6 and avg < minimum:
            minimum = avg
            label = person

    img = face_recognition.load_image_file(image)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    y1,x2,y2,x1 = face_recognition.face_locations(img)[0]
    cv2.rectangle(img, (x1, y1-30), (x2,y2), (255,0,255), 2)
    cv2.putText(img, label, (x1+6, y2+16), cv2.FONT_HERSHEY_COMPLEX, 0.6 ,(255,0,255),1)
    cv2.putText(img, str(1-minimum) + "%", (x1 + 6, y2 + 36), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 255), 1)
    while True:
        cv2.imshow("Application", img)
        key = cv2.waitKey(1)
        if key== ord('q'):
            break
    cv2.destroyAllWindows()


embeddings =  recoverEmbeddings(embeddingsFile)
testFile = "Test/billTest.jpeg"
dist = checkImage(testFile, embeddings)
figureOut(dist, testFile)

testFile = "Test/tiagoTest.jpg"
dist = checkImage(testFile, embeddings)
figureOut(dist, testFile)

testFile = "Test/elonTest.jpg"
dist = checkImage(testFile, embeddings)
figureOut(dist, testFile)