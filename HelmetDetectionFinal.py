import cv2 as cv
from ultralytics import YOLO

model = YOLO("./MuBaoHiem.pt")  # pass any model type
eyeCascade = cv.CascadeClassifier("./haarcascade_eye.xml")
faceCascade = cv.CascadeClassifier("./haarcascade_frontalface_default.xml")
body_cascade = cv.CascadeClassifier("./haarcascade_fullbody.xml")

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
stroke = 2

cap = cv.VideoCapture(0) 

def BodyDetecting(source):
    body_boxes = []

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect bodies in the frame
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around each detected body
    for (x, y, w, h) in bodies:
        cv.rectangle(frame, (x, y), (x + w, y + h), BLUE, stroke)
    
    return source, body_boxes

def FaceDetecting(source):
    face_boxes = []
    
    # Grayscale convert
    gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv.rectangle(source, (x, y), (x + w, y + h), BLUE, stroke)
        face_boxes.append((x, y, w, h))
    
    return source, face_boxes

def EyeDetecting(source):
    eye_boxes = []
    
    # Grayscale convert
    gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    
    # Detect eye
    eyes = eyeCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw bounding box
    for (x,y,w,h) in eyes:
        cv.rectangle(source, (x,y), (x+w,y+h), BLUE, stroke)
        # Store to array
        eye_boxes.append((x, y, x+w, y+h))

    return source, eye_boxes

def HelmetDetecting(source):
    detected_objects = []
    results = model.predict(source, stream=True)  
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Get class ID
            cls_id = int(box.cls[0].item())
            
            # Store to array 
            detected_objects.append((x1, y1, x2, y2, cls_id))
    
    return source, detected_objects

def HelmetChecking(source, detected_objects, eye_boxes, face_boxes, body_boxes):
    for obj in detected_objects:
        x1, y1, x2, y2, cls_id = obj
        
        if cls_id == 1:  # Class ID for helmet
            faces_in_helmet = any(
                x1 <= face_box[0] and y1 <= face_box[1] and
                (face_box[0] + face_box[2]) <= x2 and
                (face_box[1] + face_box[3]) <= y2
                for face_box in face_boxes
            )

            eyes_in_helmet = all(
                x1 <= eye_box[0] and y1 <= eye_box[1] and
                (eye_box[0] + eye_box[2]) <= x2 and
                (eye_box[1] + eye_box[3]) <= y2
                for eye_box in eye_boxes
            )

            bodies_overlap_helmet = any(
                (x1 <= bx <= x2 or x1 <= (bx + bw) <= x2) and
                (y1 <= by <= y2 or y1 <= (by + bh) <= y2)
                for (bx, by, bw, bh) in body_boxes
            )

            if faces_in_helmet or eyes_in_helmet or bodies_overlap_helmet:
                label = "Safe"
                color = GREEN
            else:
                label = "Improperly"
                color = RED

        elif cls_id == 0:  # Class ID for face
            label = "Not safe"
            color = RED
        else:
            continue  # Skip 
        
        cv.rectangle(source, (x1, y1), (x2, y2), color, stroke)
        cv.putText(source, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, stroke)
    
    return source

def ProcessingFrame(source):
    # Detect helmets and faces
    source, detected_objects = HelmetDetecting(source)
    
    # Detect faces
    source, faces_boxes = FaceDetecting(source)

    # Detect eyes
    source, eyes_boxes = EyeDetecting(source)

    # Detect eyes
    source, body_boxes = BodyDetecting(source)

    # Checking safety
    source = HelmetChecking(source, detected_objects, eyes_boxes, faces_boxes, body_boxes)

    return source

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
   
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Display the resulting frame
    process = ProcessingFrame(frame)
    cv.imshow("frame", process)

    cv.imshow("frame", frame)

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

