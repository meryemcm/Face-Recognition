import cv2
import face_recognition

imgMeryem = face_recognition.load_image_file('imgBasic/ElonMusk.jpg')
imgMeryem = cv2.cvtColor(imgMeryem, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('imgBasic/BillGates.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgMeryem)[0]
encodeElon = face_recognition.face_encodings(imgMeryem)[0]
cv2.rectangle(imgMeryem, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Elon Musk', imgMeryem)
cv2.imshow('Bill Gates', imgTest)
cv2.waitKey(0)