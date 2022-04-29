import secrets
from flask import Flask,render_template,redirect,request,Response,session
import pyrebase
import requests
import json
import cv2

import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

app=Flask(__name__)
app.secret_key="9741709968"
config={"apiKey": "AIzaSyC5tgAn81q6YrKfLJlLFO9c0bjfCEEy884",
  "authDomain": "workout-b7013.firebaseapp.com",
  "projectId": "workout-b7013",
  "storageBucket": "workout-b7013.appspot.com",
  "messagingSenderId": "909360379159",
  "appId": "1:909360379159:web:029c4e01af8e2cbb4cf352",
  "databaseURL":" ",
  "measurementId": "G-GWRFFDF5BF"}
firebase=pyrebase.initialize_app(config)

auth=firebase.auth()

@app.route("/")
def index():
    return render_template("home.html")
@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register_new_user",methods=["GET","POST"])
def register_user():
    if request.method=="POST":
        username=request.form.get("username")
        email=request.form.get("email")
        password=request.form.get("password")
        cpassword=request.form.get("confirm")
        print(username,email,password)
        request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/signupNewUser?key={0}".format(config["apiKey"])
        headers = {"content-type": "application/json; charset=UTF-8"}
        data = json.dumps({"email": email, "password": password, "returnSecureToken": True,"displayName":username})
        
        if (email!="")&(password==cpassword)&(len(password)>=4):
            try:
                request_object = requests.post(request_ref, headers=headers, data=data)
                out=request_object.json()
                print(out)
                auth.send_email_verification(out["idToken"])
                return render_template("registration_success.html")
            except:
                return render_template("registration_fail.html")
        else:
            return render_template("registration_fail.html")

@app.route("/login_user",methods=["GET","POST"])
def login_user():
    if request.method=="POST":
        email=request.form.get("email")
        password=request.form.get("password")
        user=auth.sign_in_with_email_and_password(email,password)
        user_info=auth.get_account_info(user["idToken"])
        
        session["Logged_in"]=True
        session["Registered"]=user_info["users"][0]["emailVerified"]
        session["User_name"]=user["displayName"]
        if session["Logged_in"]&session["Registered"]:
            return redirect("/start_workout")
        elif session["Logged_in"]&(not session["Registered"]):
            return render_template("login_success.html")
        return render_template("login_success.html")

def calculate_angle(a,b,c):#shoulder, elbow, wrist
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle    
    return angle

def capture_frame():
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,700)
    inputGoal = 3
    counter = 0 
    counter_r = 0
    stage = None
    stage_r = None    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:    
        while cap.isOpened():
            _,frame=cap.read()
            res = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting BGR to RGB so that it becomes easier for library to read the res
            res.flags.writeable = False #this step is done to save some memoery
            # Make detection
            results = pose.process(res) #We are using the pose estimation model 
            # Recolor back to BGR
            res.flags.writeable = True
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                # Calculate angle
                angle = calculate_angle(shoulder_l, elbow_l, wrist_l)
                
                # Get coordinates of right hand
                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                # Calculate angle
                angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                # Curl counter logic for left
                if angle > 160:
                    stage = "Down"
                if angle < 30 and stage =='Down':
                    stage="Up"
                    counter +=1

                # Curl counter logic for right
                if angle_r > 160:
                    stage_r = "Down"
                if angle_r < 30 and stage_r =='Down':
                    stage_r="Up"
                    counter_r +=1                      
            
            except:
                pass
            
            cv2.rectangle(res, (440,0), (840,60), (0,0,0), -1)
            cv2.putText(res, 'BICEP CURLS', (460,40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)

            # Render curl counter for right hand
            # Setup status box for right hand
            cv2.rectangle(res, (0,0), (70,80), (0,0,0), -1)
            # cv2.rectangle(res, (0,35), (220,80), (245,117,16), -1)
            cv2.rectangle(res, (75,0), (220,80), (0,0,0), -1)
            # Rep data
            cv2.putText(res, 'REPS', (5,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(res, str(counter_r), (10,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
            # Stage data
            cv2.putText(res, 'STAGE', (80,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(res, stage_r, (80,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
            
            
            # Render curl counter for left hand
            # Setup status box for left 
            cv2.rectangle(res, (1280-220,0), (1280-150,80), (0,0,0), -1)
            # cv2.rectangle(res, (0,35), (220,80), (245,117,16), -1)
            cv2.rectangle(res, (1280-145,0), (1280,80), (0,0,0), -1)
            # Rep data
            cv2.putText(res, 'REPS', (1280-220+5,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(res, str(counter), (1280-220+10,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
            # Stage data
            cv2.putText(res, 'STAGE', (1280-220+80,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(res, stage, (1000-220+80,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
            
            #for the instructor
            cv2.rectangle(res, (700,720-60), (1280,720), (0,0,0), -1)
            if counter > counter_r:
                cv2.putText(res, 'Do Left arm next', (720,720-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
            elif counter_r > counter:
                cv2.putText(res, 'Do Right arm next', (720,720-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
            elif counter >= inputGoal and counter_r >= inputGoal:
                cv2.putText(res, 'GOOD JOB', (720,960-60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2, cv2.LINE_AA)
            # Render detections
            mp_drawing.draw_landmarks(res, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            #do your processing here
            _,buffer=cv2.imencode(".jpg",res)
            res=buffer.tobytes()
            yield(b' --frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+res+b'\r\n')
            
            if int(counter) >= int(inputGoal) and int(counter_r) >= int(inputGoal):
                img = cv2.imread("./assets/Workout Completed.jpg", cv2.IMREAD_COLOR)
                cv2.imshow("Hello", img)
                break

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
    cap.release()
    cv2.destroyAllWindows()
    # /////////////////////////////////////////////////

    cap = cv2.VideoCapture('./assets/Countdown5.mp4')
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, res1 = cap.read()
        if ret == True:
            _,buffer=cv2.imencode(".jpg",res1)
            res1=buffer.tobytes()
            yield(b' --frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+res1+b'\r\n')
        else: 
            break
    cap.release()
    cv2.destroyAllWindows()
        
        
    # ////////////////////////////////////////////////
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,700)
    
    counter = 0 
    counter_r = 0
    stage = None
    stage_r = None
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            # Recolor image1 to RGB
            image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting BGR to RGB so that it becomes easier for library to read the image1
            image1.flags.writeable = False #this step is done to save some memoery
            # Make detection
            results = pose.process(image1) #We are using the pose estimation model 
            # Recolor back to BGR
            image1.flags.writeable = True
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                # Calculate angle of left full
                angle = calculate_angle(hip_l, knee_l, ankle_l)
                
                
                hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                # Calculate angle
                angle_r = calculate_angle(hip_r, knee_r, ankle_r)
                
                # Curl counter logic for left
                if angle > 140:
                    stage = "Down"
                if angle < 120 and stage =='Down':
                    stage="Up"
                    counter +=1
                    print("Left : ",counter)

                # Curl counter logic for right
                if angle_r > 140:
                    stage_r = "Down"
                if angle_r < 120 and stage_r =='Down':
                    stage_r="Up"
                    counter_r +=1
                    print("Right : ",counter_r)                       
            
            except:
                pass
            cv2.rectangle(image1, (300,0), (600,60), (0,0,0), -1)
            cv2.putText(image1, 'HIGH KNEES', (320,40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
            # Render curl counter for right hand
            # Setup status box for right hand
            cv2.rectangle(image1, (0,0), (70,80), (0,0,0), -1)
            # cv2.rectangle(image1, (0,35), (220,80), (245,117,16), -1)
            cv2.rectangle(image1, (75,0), (220,80), (0,0,0), -1)
            # Rep data
            cv2.putText(image1, 'REPS', (5,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image1, str(counter_r), (10,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
            # Stage data
            cv2.putText(image1, 'STAGE', (80,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image1, stage_r, (80,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
            
            
            # Render curl counter for left hand
            # Setup status box for left 
            cv2.rectangle(image1, (1000-220,0), (1000-150,80), (0,0,0), -1)
            # cv2.rectangle(image1, (0,35), (220,80), (245,117,16), -1)
            cv2.rectangle(image1, (1000-145,0), (1000,80), (0,0,0), -1)
            # Rep data
            cv2.putText(image1, 'REPS', (1000-220,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image1, str(counter), (1000-220,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
            # Stage data
            cv2.putText(image1, 'STAGE', (1000-220+75,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image1, stage, (1000-220+70,65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1, cv2.LINE_AA)
            
            #for the instructor
            cv2.rectangle(image1, (600,700-60), (1000,700), (0,0,0), -1)
            if counter > counter_r:
                cv2.putText(image1, 'Do Left Leg next', (630,700-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
            elif counter_r > counter:
                cv2.putText(image1, 'Do Right Leg next', (630,700-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
            elif counter == inputGoal and counter_r == inputGoal:
                cv2.putText(image1, 'GOOD JOB', (540,960-60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2, cv2.LINE_AA)
                
            # Render detections
            mp_drawing.draw_landmarks(image1, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
                
            _,buffer=cv2.imencode(".jpg",image1)
            image1=buffer.tobytes()
            yield(b' --frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+image1+b'\r\n')
            
            
            if int(counter) >= int(inputGoal) and int(counter_r) >= int(inputGoal):
                img = cv2.imread("./assets/Workout Completed.jpg", cv2.IMREAD_COLOR)
                cv2.imshow("Hello", img)
                break

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
    cap.release()
    cv2.destroyAllWindows()
    
    # //////////////////////////////////////////
    cap = cv2.VideoCapture('./assets/Workout Completed.jpg')
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, res1 = cap.read()
        if ret == True:
            _,buffer=cv2.imencode(".jpg",res1)
            res1=buffer.tobytes()
            yield(b' --frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+res1+b'\r\n')
        else: 
            break
    cap.release()
    cv2.destroyAllWindows()

@app.route("/start_workout",methods=["GET","POST"])
def workout():
    if ("Logged_in" in session)&("Registered" in session):
        if session["Logged_in"]&~(session["Registered"]):
            return render_template("verify_first.html")
        else:
            return render_template("workout.html")
            
    elif "Logged_in" not in session:
        return render_template("please_register.html")
    else:
        return render_template("workout.html")
@app.route("/video")
def video():
    if ("Logged_in" not in session)&("Registered" not in session):
        return render_template("please_login.html")
    else:

        return Response(capture_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/logout")
def logout():
    session.pop("Logged_in",None)
    session.pop("User_name",None)
    session.pop("Registered",None)
    return redirect("/")
    
if __name__=="__main__":
    app.run(debug=True)