import numpy as np
import cv2
import os.path
import sys



def detect_faces():

    # initialization
    cascades_path = "/usr/share/opencv/haarcascades/"
    face_path = os.path.join(cascades_path, "haarcascade_frontalface_default.xml")
    assert os.path.exists(face_path)
    # Set up cascades
    face_cascade = cv2.CascadeClassifier(face_path)

    # create a VideoCapture object and set it to camera
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # face detection
        faces = face_cascade.detectMultiScale(frame, 1.1,3, minSize=(20, 150))

        # Display the resulting frame
        cv2.imshow('frame', frame)

        #check if face detected
        if (len(faces) != 0):
            # Display the resulting frame
            for (x, y, w, h) in faces:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return  faces

def track_faces(faces):

    cap = cv2.VideoCapture(0)
    #getting the face surrounding Box
    x=faces[0][0]
    y=faces[0][1]
    w=faces[0][2]
    h=faces[0][3]
    track_window = (x, y, w, h)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        cv2.imshow('frame', frame)
        # set up the ROI for tracking
        roi = frame[y:y + h, x:x + w]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 90], 1)

            # apply meanshift to get the new locations

            ret, track_window = cv2.CamShift(dst, track_window, term_crit)

            # Draw it on image

            pts = cv2.boxPoints(ret)

            pts = np.int0(pts)

            img2 = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            cv2.imshow('frame', img2)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def main():
    #initializon of the variable
    faces=()
    #call face detection function
    faces = detect_faces()
    # if face detected go to tracking
    if len(faces)>0:
        track_faces(faces)
    #Ask for repeat face detection or quite
    else:
       print "No face detected !! press R to repeat the face detection Q to quite"

       name = raw_input("Enter your Choise: ")
       if name == 'r':
           faces = ()
           faces = detect_faces()
       elif name=='q':

           cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

