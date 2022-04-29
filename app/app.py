from flask import Flask,render_template,redirect,request,Response
import pyrebase
import cv2
app=Flask(__name__)

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
        
        email=request.form.get("email")
        password=request.form.get("password")
        cpassword=request.form.get("confirm")

        
        if (email!="")&(password==cpassword)&(len(password)>=4):
            try:
                user=auth.create_user_with_email_and_password(email,password)
                
                auth.send_email_verification(user["idToken"])
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
        print(user)
        return render_template("login_success.html")



def capture_frame():
    vid=cv2.VideoCapture(0)
    
    while True:
        _,frame=vid.read()

        #do your processing here
        _,buffer=cv2.imencode(".jpg",frame)

        frame=buffer.tobytes()
        
        yield(b' --frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

@app.route("/start_workout",methods=["GET","POST"])
def workout():
    
    return render_template("workout.html")
@app.route("/video")
def video():
    return Response(capture_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=="__main__":
    app.run(debug=True)