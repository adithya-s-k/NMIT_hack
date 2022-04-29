import secrets
from flask import Flask,render_template,redirect,request,Response,session
import pyrebase
import requests
import json
import cv2
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