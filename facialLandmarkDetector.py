
import dlib,cv2
import numpy as np
from renderFace import renderFace
import json

def writeLandmarksToFile(landmarks, landmarksFileName):
    result = [{'x': p.x, 'y': p.y} for p in landmarks.parts()]
    with open(landmarksFileName, 'w') as fout:
        json.dump(result, fout, indent=2)

      # f.write("x: {}, y: {}\n".format(int(p.x), int(p.y)))
      # f.write("%s %s\n" %(int(p.x),int(p.y)))

    fout.close()

# Landmark model location
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
# Get the face detector
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

# Read image
#imageFilename = "hillary_clinton.jpg"
cam = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 10.0, (640,480))


while 1:

  ret, img = cam.read()
#im= cv2.imread(imageFilename)
# landmarks will be stored in results/family_i.txt
  landmarksBasename = "results/faces"

# Detect faces in the image
  faceRects = faceDetector(img, 0)
  print("Number of faces detected: ",len(faceRects))

# List to store landmarks of all detected faces
  landmarksAll = []

# Loop over all detected face rectangles
  for i in range(0, len(faceRects)):
    newRect = dlib.rectangle(int(faceRects[i].left()),int(faceRects[i].top()),
      int(faceRects[i].right()),int(faceRects[i].bottom()))

  # For every face rectangle, run landmarkDetector
    landmarks = landmarkDetector(img, newRect)

    # Print number of landmarks
    if i==0:
      print("Number of landmarks",len(landmarks.parts()))

    # Store landmarks for current face
    landmarksAll.append(landmarks)

    # Draw landmarks on face
    renderFace(img, landmarks)

    landmarksFileName = landmarksBasename +"_"+ str(i)+ ".json"
    print("Saving landmarks to", landmarksFileName)

    # Write landmarks to disk
    writeLandmarksToFile(landmarks, landmarksFileName)

    out.write(img)

    cv2.imshow("Facial Landmark detector", img)

    k= cv2.waitKey(10) & 0xff
    if k==27:
        break

        cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()