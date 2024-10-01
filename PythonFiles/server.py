import mediapipe as mp
import numpy as np
import cv2
from flask import Flask
from flask_cors import CORS, cross_origin
import datetime
from mediapipe.framework.formats import landmark_pb2
import joblib
from cvzone.PoseModule import PoseDetector
import math

# Initializing flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
  
# Route for seeing a data
@app.route('/shoulderpress')
def shoulderpress():

    cap = cv2.VideoCapture(0)
    detector = PoseDetector(detectionCon=0.7, trackCon=0.7)

    # Creating Angle finder class
    class angleFinder:
        def __init__(self, lmlist, p1, p2, p3, p4, p5, p6, drawPoints):
            self.lmlist = lmlist
            self.p1 = p1
            self.p2 = p2
            self.p3 = p3
            self.p4 = p4
            self.p5 = p5
            self.p6 = p6
            self.drawPoints = drawPoints

        # finding angles
        def angle(self):
            if len(self.lmlist) != 0:
                point1 = self.lmlist[self.p1]
                point2 = self.lmlist[self.p2]
                point3 = self.lmlist[self.p3]
                point4 = self.lmlist[self.p4]
                point5 = self.lmlist[self.p5]
                point6 = self.lmlist[self.p6]

                print(f"point1: {point1}")
                print(f"point2: {point2}")
                print(f"point3: {point3}")
                print(f"point4: {point4}")
                print(f"point5: {point5}")
                print(f"point6: {point6}")

                if len(point1) >= 2 and len(point2) >= 2 and len(point3) >= 2 and len(point4) >= 2 and len(point5) >= 2 and len(
                        point6) >= 2:
                    x1, y1 = point1[:2]
                    x2, y2 = point2[:2]
                    x3, y3 = point3[:2]
                    x4, y4 = point4[:2]
                    x5, y5 = point5[:2]
                    x6, y6 = point6[:2]

                    # calculating angle for left and right hands
                    leftHandAngle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
                    rightHandAngle = math.degrees(math.atan2(y6 - y5, x6 - x5) - math.atan2(y4 - y5, x4 - x5))

                    leftHandAngle = int(np.interp(leftHandAngle, [-170, 180], [100, 0]))
                    rightHandAngle = int(np.interp(rightHandAngle, [-50, 20], [100, 0]))

                    # drawing circles and lines on selected points
                    if self.drawPoints:
                        cv2.circle(img, (x1, y1), 10, (0, 255, 255), 5)
                        cv2.circle(img, (x1, y1), 15, (0, 255, 0), 6)
                        cv2.circle(img, (x2, y2), 10, (0, 255, 255), 5)
                        cv2.circle(img, (x2, y2), 15, (0, 255, 0), 6)
                        cv2.circle(img, (x3, y3), 10, (0, 255, 255), 5)
                        cv2.circle(img, (x3, y3), 15, (0, 255, 0), 6)
                        cv2.circle(img, (x4, y4), 10, (0, 255, 255), 5)
                        cv2.circle(img, (x4, y4), 15, (0, 255, 0), 6)
                        cv2.circle(img, (x5, y5), 10, (0, 255, 255), 5)
                        cv2.circle(img, (x5, y5), 15, (0, 255, 0), 6)
                        cv2.circle(img, (x6, y6), 10, (0, 255, 255), 5)
                        cv2.circle(img, (x6, y6), 15, (0, 255, 0), 6)

                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 4)
                        cv2.line(img, (x4, y4), (x5, y5), (0, 0, 255), 4)
                        cv2.line(img, (x5, y5), (x6, y6), (0, 0, 255), 4)
                        cv2.line(img, (x1, y1), (x4, y4), (0, 0, 255), 4)

                    return [leftHandAngle, rightHandAngle]

    # defining some variables
    counter = 0
    direction = 0
    # defining some variables
    counter = 0
    direction = 0

    while True:
        ret, img = cap.read()
        img = cv2.resize(img, (640, 480))

        detector.findPose(img, draw=0)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=0, draw=False)

        angle1 = angleFinder(lmList, 11, 13, 15, 12, 14, 16, drawPoints=True)
        hands = angle1.angle()
        left, right = hands[0:]

        # Counting number of shoulder ups
        if left >= 90 and right >= 90:
            if direction == 0:
                counter += 0.5
                direction = 1
        if left <= 70 and right <= 70:
            if direction == 1:
                counter += 0.5
                direction = 0

        # putting scores on the screen
        cv2.rectangle(img, (0, 0), (120, 120), (255, 0, 0), -1)
        cv2.putText(img, str(int(counter)), (1, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 6)

        # Converting values for rectangles
        leftval = np.interp(left, [0, 100], [400, 200])
        rightval = np.interp(right, [0, 100], [400, 200])

        # For color changing
        value_left = np.interp(left, [0, 100], [0, 100])
        value_right = np.interp(right, [0, 100], [0, 100])

        # Drawing right rectangle and putting text
        cv2.putText(img, 'R', (24, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 5)
        cv2.rectangle(img, (8, 200), (50, 400), (0, 255, 0), 5)
        cv2.rectangle(img, (8, int(rightval)), (50, 400), (255, 0, 0), -1)

        # Drawing right rectangle and putting text
        cv2.putText(img, 'L', (604, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 5)
        cv2.rectangle(img, (582, 200), (632, 400), (0, 255, 0), 5)
        cv2.rectangle(img, (582, int(leftval)), (632, 400), (255, 0, 0), -1)

        if value_left > 70:
            cv2.rectangle(img, (582, int(leftval)), (632, 400), (0, 0, 255), -1)

        if value_right > 70:
            cv2.rectangle(img, (8, int(rightval)), (50, 400), (0, 0, 255), -1)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return "Done"



@app.route('/bicepcurl')
def bicepcurl():
    def calc_angle(a,b,c): # 3D points
        ''' Arguments:
            a,b,c -- Values (x,y,z, visibility) of the three points a, b and c which will be used to calculate the
                    vectors ab and bc where 'b' will be 'elbow', 'a' will be shoulder and 'c' will be wrist.
            
            Returns:
            theta : Angle in degress between the lines joined by coordinates (a,b) and (b,c)
        '''
        a = np.array([a.x, a.y])#, a.z])    # Reduce 3D point to 2D
        b = np.array([b.x, b.y])#, b.z])    # Reduce 3D point to 2D
        c = np.array([c.x, c.y])#, c.z])    # Reduce 3D point to 2D

        ab = np.subtract(a, b)
        bc = np.subtract(b, c)
        
        theta = np.arccos(np.dot(ab, bc) / np.multiply(np.linalg.norm(ab), np.linalg.norm(bc)))     # A.B = |A||B|cos(x) where x is the angle b/w A and B
        theta = 180 - 180 * theta / 3.14    # Convert radians to degrees
        return np.round(theta, 2)



    mp_drawing = mp.solutions.drawing_utils     # Connecting Keypoints Visuals
    mp_pose = mp.solutions.pose                 # Keypoint detection model
    left_flag = None     # Flag which stores hand position(Either UP or DOWN)
    left_count = 0       # Storage for count of bicep curls
    right_flag = None
    right_count = 0

    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) # Lnadmark detection model instance
    while cap.isOpened():
        _, frame = cap.read()

        # BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Convert BGR frame to RGB
        image.flags.writeable = False
        
        # Make Detections
        results = pose.process(image)                       # Get landmarks of the object in frame from the model

        # Back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # Convert RGB back to BGR

        try:
            # Extract Landmarks
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Calculate angle
            left_angle = calc_angle(left_shoulder, left_elbow, left_wrist)      #  Get angle 
            right_angle = calc_angle(right_shoulder, right_elbow, right_wrist)

            # Visualize angle
            cv2.putText(image,\
                    str(left_angle), \
                        tuple(np.multiply([left_elbow.x, left_elbow.y], [640,480]).astype(int)),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
            cv2.putText(image,\
                    str(right_angle), \
                        tuple(np.multiply([right_elbow.x, right_elbow.y], [640,480]).astype(int)),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
        
            # Counter 
            if left_angle > 160:
                left_flag = 'down'
            if left_angle < 50 and left_flag=='down':
                left_count += 1
                left_flag = 'up'

            if right_angle > 160:
                right_flag = 'down'
            if right_angle < 50 and right_flag=='down':
                right_count += 1
                right_flag = 'up'
            
        except:
            pass

        # Setup Status Box
        cv2.rectangle(image, (0,0), (1024,73), (10,10,10), -1)
        cv2.putText(image, 'Left=' + str(left_count) + '    Right=' + str(right_count),
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        # Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('MediaPipe feed', image)

        k = cv2.waitKey(30) & 0xff  # Esc for quiting the app
        if k==27:
            break
        elif k==ord('r'):       # Reset the counter on pressing 'r' on the Keyboard
            left_count = 0
            right_count = 0

    cap.release()
    cv2.destroyAllWindows()


    return "Done"
      
@app.route('/legextensions')
def legextensions():

    def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle
    # webcam input
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None

    """width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('your_video.mp4', fourcc, 10.0, size, True)"""

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                
                # Get coordinates
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
            
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                angle_knee = calculate_angle(hip, knee, ankle) #Knee joint angle
                
                angle_hip = calculate_angle(shoulder, hip, knee)
                hip_angle = 180-angle_hip
                knee_angle = 180-angle_knee
                
                # Visualize angle
                """cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )"""
                        
                    
                cv2.putText(image, str(angle_knee), 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 121, 66), 2, cv2.LINE_AA
                                    )
                
                """cv2.putText(image, str(angle_hip), 
                            tuple(np.multiply(hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )"""
                
                # Curl counter logic
                if angle_knee > 169:
                    stage = "UP"
                if angle_knee <= 90 and stage =='UP':
                    stage="DOWN"
                    counter +=1
                    print(counter)
            except:
                pass
            
            # Render squat counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        #out.release()
        cv2.destroyAllWindows()
        
    #destroyAllWindows()



    return ("Done")

@app.route('/lunges')
def lunges():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles
    PoseLandmark = mp.solutions.pose.PoseLandmark

    ####### FIXED VARIABLES ##########
    font = cv2.FONT_HERSHEY_TRIPLEX
    webcam_dimensions = [640, 480]

    # Colours
    black = (0, 0, 0)
    white = (255, 255, 255)
    green = (97,250,2)
    red = (19,3,252)
    grey = (131, 133, 131)
    light_blue = (237, 215, 168)

    ####### FUNCTIONS ##########
    def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle 

    def render_status_box(counter, stage, fontScale, form):
        cv2.rectangle(image, (0,0), (235,75), light_blue, -1)
        cv2.rectangle(image, (300, 0), (640, 75), white, -1)
        cv2.putText(image, 'REPS', (15,12), font, 0.5, black, 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10,60), font, fontScale, white, 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (105,12), font, 0.5, black, 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60,60), font, 2, white, 2, cv2.LINE_AA)
        
        if form == "correct":
            cv2.putText(image, form, 
                    (330, 50), 
                    font, 2, green, 2, cv2.LINE_AA)
        else:
            cv2.putText(image, form, 
                    (315, 50), 
                    font, 2, red, 2, cv2.LINE_AA)


    ######### START CHECKING EXERCISE ###########
    lunges_coordinates = {(11, 23),
                        (12, 24),
                        (23, 25),
                        (23, 24),
                        (25, 27),
                        (24, 26),
                        (26, 28)}
    lunges_connections = frozenset(lunges_coordinates)

    cap = cv2.VideoCapture(0)

    # Lunge counter variables
    counter = 0 
    stage = None
    form = None

    # function to convert angles to a value easy to use for lunge logic
    def validate_angle(ang) :
        n = (ang-170)*0.05
        return (n >= 0 and n <= 1)

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cv2.namedWindow('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee =  [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee =  [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                
                # Calculate angle
                right_knee_angle = calculate_angle(right_hip,right_knee,right_ankle)
                left_knee_angle = calculate_angle(left_hip,left_knee,left_ankle)
                body_straight = calculate_angle(left_shoulder,left_hip,left_knee)
                
                # Visualize angle
                cv2.putText(image, str(round(right_knee_angle, 2)), 
                            tuple(np.multiply(right_knee, webcam_dimensions).astype(int)), 
                            font, 0.5, white, 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(round(left_knee_angle, 2)), 
                            tuple(np.multiply(left_knee, webcam_dimensions).astype(int)), 
                            font, 0.5, white, 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(body_straight), 
                            tuple(np.multiply(left_hip, webcam_dimensions).astype(int)), 
                            font, 0.5, white, 2, cv2.LINE_AA
                                    )
                # Lunge counter logic
                if (left_knee_angle > 90 and right_knee_angle < 90) or (right_knee_angle > 90 and left_knee_angle < 90):
                    stage = "front"
                    form = "correct"
                else : 
                    form = "incorrect"
                    
                if validate_angle(body_straight) :
                    form = "correct"
                    
                if not validate_angle(left_knee_angle) and validate_angle(right_knee_angle) and stage == 'front' :
                    form = "incorrect"
                    
                if validate_angle(left_knee_angle) and validate_angle(right_knee_angle) and stage == 'front' :
                    stage="back"
                    form = "correct"
                    counter +=1
                        
            except:
                pass

            # Render detections
            if form == "incorrect":
                mp_drawing.draw_landmarks(image, results.pose_landmarks, lunges_connections,
                                        mp_drawing.DrawingSpec(color=grey, thickness=0, circle_radius=0), 
                                        mp_drawing.DrawingSpec(color=red, thickness=2, circle_radius=2))
                
            elif form == "correct":
                mp_drawing.draw_landmarks(image, results.pose_landmarks, lunges_connections,
                                        mp_drawing.DrawingSpec(color=grey, thickness=0, circle_radius=0), 
                                        mp_drawing.DrawingSpec(color=green, thickness=2, circle_radius=2))
                
        
            if counter < 10:
                render_status_box(counter, stage, 2, form)
            else:
                render_status_box(counter, stage, 1, form)                                           

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        
        cap.release()
        cv2.destroyAllWindows()

    joblib.dump(lunges, 'classifier.joblib')
    return "lunges"


@app.route('/pushup')
def pushup():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles
    PoseLandmark = mp.solutions.pose.PoseLandmark

    ####### FIXED VARIABLES ##########
    font = cv2.FONT_HERSHEY_TRIPLEX
    webcam_dimensions = [640, 480]

    # Colours
    black = (0, 0, 0)
    white = (255, 255, 255)
    green = (97,250,2)
    red = (19,3,252)
    grey = (131, 133, 131)
    light_blue = (237, 215, 168)

    ####### FUNCTIONS ##########
    def calculate_angles(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle 

    def render_status_boxs(counter, stage, fontScale, form):
        cv2.rectangle(image, (0,0), (235,75), light_blue, -1)
        cv2.rectangle(image, (300, 0), (640, 75), white, -1)
        cv2.putText(image, 'REPS', (15,12), font, 0.5, black, 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10,60), font, fontScale, white, 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (105,12), font, 0.5, black, 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60,60), font, 2, white, 2, cv2.LINE_AA)
        
        if form == "correct":
            cv2.putText(image, form, 
                    (330, 50), 
                    font, 2, green, 2, cv2.LINE_AA)
        else:
            cv2.putText(image, form, 
                    (315, 50), 
                    font, 2, red, 2, cv2.LINE_AA)


    ######### START CHECKING EXERCISE ###########
    pushup_coordinates = {(11, 23),
                        (12, 24),
                        (23, 25),
                        (24, 26)}
    pushup_connections = frozenset(pushup_coordinates)

    cap = cv2.VideoCapture(0)

    # Push-up counter variables
    counter = 0 
    stage = None
    form = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cv2.namedWindow('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angles
                left_hip_angle = calculate_angles(left_shoulder, left_hip, left_ankle)
                left_elbow_angle = calculate_angles(left_shoulder, left_elbow, left_wrist)
                
                right_hip_angle = calculate_angles(right_shoulder, right_hip, right_ankle)
                right_elbow_angle = calculate_angles(right_shoulder, right_elbow, right_wrist)
                
                # Visualize angle
                cv2.putText(image, str(round(left_hip_angle, 2)), 
                            tuple(np.multiply(left_hip, webcam_dimensions).astype(int)), 
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, white, 2, cv2.LINE_AA)
                
                cv2.putText(image, str(round(right_hip_angle, 2)), 
                            tuple(np.multiply(left_hip, webcam_dimensions).astype(int)), 
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, white, 2, cv2.LINE_AA)
            except:
                pass

            
            # Form data and pushup counter logic
            if left_hip_angle < 155 or right_hip_angle < 155:
                form = "incorrect"
            elif left_hip_angle > 155 or right_hip_angle > 155:
                form = "correct"
            if left_elbow_angle < 70 or right_elbow_angle < 70:
                stage="down"
            elif (left_elbow_angle > 160 or right_elbow_angle > 160) and stage=="down":
                stage="up"
                counter+=1
                
            
            # Render detections
            if form == "incorrect":
                mp_drawing.draw_landmarks(image, results.pose_landmarks, pushup_connections,
                                        mp_drawing.DrawingSpec(color=grey, thickness=0, circle_radius=0), 
                                        mp_drawing.DrawingSpec(color=red, thickness=2, circle_radius=2))
                
            elif form == "correct":
                mp_drawing.draw_landmarks(image, results.pose_landmarks, pushup_connections,
                                        mp_drawing.DrawingSpec(color=grey, thickness=0, circle_radius=0), 
                                        mp_drawing.DrawingSpec(color=green, thickness=2, circle_radius=2))
                
        
            if counter < 10:
                render_status_boxs(counter, stage, 2, form)
            else:
                render_status_boxs(counter, stage, 1, form)
                
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    joblib.dump(pushup, 'classifier.joblib')
    return "Done"


# Running app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
