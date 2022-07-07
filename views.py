from django.shortcuts import render,HttpResponseRedirect
import MySQLdb
import datetime
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import simplejson as json
from datetime import date
from datetime import datetime
import datetime
import webbrowser
import math, random 
import os
import glob
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err
def compare_images(imageA, imageB, title):
    out=0
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    
    if(s>=.5 and s<=1):
        out=1
        msg="detected"
    else:
        msg="notdetected"
    fig = plt.figure(title)
    plt.suptitle("Similarity %s,Value %.2f"%(msg,s))
	# show first image
    # 
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
	# show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
	# show the images
    plt.show()
    return out

def sendsms(ph,msg):
    sendToPhoneNumber= "+91"+ph
    userid = "2000022557"
    passwd = "54321@lcc"
    url = "http://enterprise.smsgupshup.com/GatewayAPI/rest?method=sendMessage&send_to=" + sendToPhoneNumber + "&msg=" + msg + "&userid=" + userid + "&password=" + passwd + "&v=1.1&msg_type=TEXT&auth_scheme=PLAIN"
    # contents = urllib.request.urlopen(url)
    # webbrowser.open(url)

def generateOTP() :   
    # Declare a digits variable   
    # which stores all digits  
    digits = "0123456789"
    OTP = "" 
   # length of password can be chaged 
   # by changing value in range 
    for i in range(4) : 
        OTP += digits[math.floor(random.random() * 10)] 
        print(i)
    return OTP 

# Create your views here.

db = MySQLdb.connect('localhost','root','','glycoma')
c = db.cursor()

def AdminHome(request):
    return render(request,'AdminHome.html') 

def CommonHome(request):
    return render(request,'index.html')

def DoctorHome(request):
    return render(request,'DoctorHome.html')

def PatientHome(request):
    return render(request,'PatientHome.html')

def CustomerSignup(request):
    if request.POST:
        cname = request.POST.get("cname")
        address = request.POST.get("address")
        age = request.POST.get("age")
        adhar = request.POST.get("adhar")
        blood = request.POST.get("bg")
        gender = request.POST.get("gender")
        Email = request.POST.get("Email")
        mobile = request.POST.get("mobile")
        password = request.POST.get("Password")
        qry = "insert into patient_reg(name,email,age,gender,phone,address,adharno,blood,password) values('"+ str(cname) +"','"+ str(Email) +"','"+str(age)+"','"+ str(gender) +"','"+ str(mobile) +"','"+str(address)+"','"+ str(adhar) +"','"+str(blood)+"','"+ str(password) +"')"
        qr = "insert into login values('"+ str(Email) +"','"+ str(password) +"','Patient')"
        c.execute(qry)
        c.execute(qr)
        db.commit()
        c.execute("select max(id) from patient_reg")
        pid = c.fetchone()
        patid = pid[0]
        msg = "Greetings.. Your unique ID is " + str(patid)
        sendsms(mobile,msg)
        return HttpResponseRedirect('/SignIn/')
    return render(request,'CustomerSignup.html')


def DoctorSignup(request):
    c.execute("select * from hospital_reg")
    data=c.fetchall()
    if request.POST:
        cname = request.POST.get("cname")
        address = request.POST.get("address")
        sprc = request.POST.get("sprc")
        hosp = request.POST.get("hosp")
        Email = request.POST.get("Email")
        mobile = request.POST.get("mobile")
        password = request.POST.get("Password")
        qry = "insert into doctor_reg(name,email,phone,department,spacialization,password,hos_name,status) values('"+ str(cname) +"','"+ str(Email) +"','"+str(mobile)+"','"+str(address) +"','"+ str(sprc) +"','"+str(password)+"','"+ str(hosp) +"','Registered')"
        qr = "insert into login values('"+ str(Email) +"','"+ str(password) +"','Doctor')"
        c.execute(qry)
        c.execute(qr)
        db.commit()
        return HttpResponseRedirect('/SignIn/')
    return render(request,'DoctorSignup.html',{"data":data})

def SignIn(request):  
    request.session['username']=""
    request.session['NAME']=""
    request.session['uid']=""
    request.session['cid']=""
    request.session['sid']=""
    msg=""
    if request.POST:
        email = request.POST.get("email")
        password = request.POST.get("password")
        c.execute("select * from login where uname='"+ email +"' and pass='"+ password +"'")
        ds = c.fetchone()
        request.session['username']=email
        if ds is not None:
            if ds[2] == 'Admin':
                return HttpResponseRedirect('/AdminHome/')
            elif ds[2] == 'Patient':
                c.execute("select * from patient_reg where email='"+email+"' and password='"+password+"'")
                ds = c.fetchone()
                request.session['pid'] = ds[0]
                request.session['NAME'] = ds[1]
                return HttpResponseRedirect('/PatientHome/')
            elif ds[2] == 'Doctor':
                c.execute("select * from doctor_reg where email='"+email+"' and password='"+password+"'")
                ds = c.fetchone()
                request.session['did'] = ds[0]
                return HttpResponseRedirect('/DoctorHome/')
        else:
            msg = "Incorrect username or password"
    return render(request,'Login.html',{"msg":msg}) 


def AdminViewCustomers(request):
    data = ""
    c.execute("select * from patient_reg")
    data=c.fetchall() 
    return render (request,"AdminViewCustomers.html",{"data":data})

def AdminAddHospital(request):
    msg = ""
    if request.POST:
        name = request.POST.get("hname")
        adrs = request.POST.get("address")
        phone = request.POST.get("phone")
        email = request.POST.get("email")
        qry = "insert into hospital_reg(hname,address,phone,email) values('"+ name +"','"+ adrs +"','"+ phone +"','"+ email +"')"
        c.execute(qry)
        db.commit()
        msg = "Hospital Details Added Successfully."
    return render(request,'AdminAddHospital.html',{"msg":msg})

def AdminViewHospital(request):
    data = ""
    c.execute("select * from hospital_reg")
    data=c.fetchall()
    if request.GET:
        hid = request.GET.get("id")
        c.execute("delete from hospital_reg where hid = '"+str(hid)+"'")
        db.commit()
        return HttpResponseRedirect("/AdminViewHospital/")
    return render (request,"AdminViewHospital.html",{"data":data})

def AdminViewDoctors(request):
    data = ""
    c.execute("select * from doctor_reg  where status = 'Registered'")
    data=c.fetchall() 
    if request.GET:
        rid = request.GET.get("id")
        st = request.GET.get("st")
        if st == 'Accept':
            c.execute("update doctor_reg set status = '"+st+"' where id = '"+str(rid)+"'")
            db.commit()
            return HttpResponseRedirect("/AdminViewDoctors/")
        else:
            c.execute("delete from doctor_reg where id = '"+str(rid)+"'")
            db.commit()
            return HttpResponseRedirect("/AdminViewDoctors/")
    return render (request,"AdminViewDoctors.html",{"data":data})

def AdminViewApprovedDoctors(request):
    data = ""
    c.execute("select * from doctor_reg where status = 'Accept'")
    data=c.fetchall() 
    return render (request,"AdminViewApprovedDoctors.html",{"data":data})

def AdminViewFeedback(request):
    data = ""
    c.execute("select * from patient_reg inner join feedback on patient_reg.id = feedback.userid")
    data=c.fetchall() 
    return render (request,"AdminViewFeedback.html",{"data":data})

def CustomerSearchDoctor(request):
    c.execute("select * from hospital_reg")
    data = c.fetchall()
    data1=""
    if request.POST:
        a = request.POST.get("dist")
        c.execute("select * from doctor_reg where hos_name = '"+a+"'")
        data1 = c.fetchall()
    if request.GET:
        did = request.GET.get("id")
        request.session["did"] = did
        return HttpResponseRedirect("/PatientUploadImage/")
    return render (request,"CustomerSearchDoctor.html",{"data":data,"data1":data1})

def PatientUploadImage(request):
    did = request.GET.get("id")
    pid = request.session["pid"]
    msg=""
    if request.POST:
        a = request.POST.get("det")
        if request.FILES.get("file"):
            myfile=request.FILES.get("file")
            fs=FileSystemStorage()
            filename=fs.save(myfile.name , myfile)
            uploaded_file_url = fs.url(filename)
            # from django.contrib.staticfiles.storage import staticfiles_storage
            # z=staticfiles_storage.url('media/LMM2_orig.jpg')
            
            print(os.path.join(BASE_DIR,''))
            filedata=os.scandir(os.path.join(BASE_DIR,'images/dermquest'))
            
           
            for item in filedata:
                print(item.name)
            print("########################################################################")
            original = cv2.imread("D:/MAIN PROJECT/Glycoma/images/dermquest/"+filename)
            print(uploaded_file_url)
            try:

                contrast = cv2.imread('D:/MAIN PROJECT/Glycoma/GlycomaApp/static'+uploaded_file_url)
                original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
                ss=compare_images(original, contrast, "Original vs. Contrast")
            
                msg="Detected"
               
            except:
                msg="Not detected"
            # if ss==1 :
            #     break
            
            # fig = plt.figure("Images")
            # images = ("Original", original), ("Contrast", contrast)
            # # loop over the images
            # for (i, (name, image)) in enumerate(images):
            #     # show the image
            #     ax = fig.add_subplot(1, 2, i + 1)
            #     ax.set_title(name)
            #     plt.imshow(image, cmap = plt.cm.gray)
            #     plt.axis("off")
            # # show the figure
            # plt.show()
            # compare the images
            #compare_images(original, original, "Original vs. Original")
            
        c.execute("insert into consultation(pid,did,cdetails,image,status)values('"+str(pid)+"','"+str(did)+"','"+str(a)+"','"+str(uploaded_file_url)+"','Upload')") 
        db.commit()  
    return render (request,"PatientUploadImage.html",{"msg":msg})


def DoctorViewConsultationRequest(request):
    data = ""
    c.execute("select * from patient_reg inner join consultation on patient_reg.id = consultation.pid where status = 'Upload'")
    data=c.fetchall() 
    if request.GET:
        coid = request.GET.get("id")
        st = request.GET.get("st")
        fo = request.GET.get("fo")
        if st == "Reject":
            c.execute("delete from consultation where cid = '"+coid+"'")
            db.commit()
            msg = "Your consultation request rejected"
            sendsms(fo,msg)
    return render (request,"DoctorViewConsultationRequest.html",{"data":data})


####################################################################
# try:
#     #from __future__ import division, print_function
#     import sys
#     import os
#     import glob
#     import re
#     from pathlib import Path
#     from io import BytesIO
#     import base64
#     import requests

#     # Import fast.ai Library
#     from fastai import *
#     from fastai.vision import *

#     # Flask utils
#     from flask import Flask, redirect, url_for, render_template, request
#     from PIL import Image as PILImage

#     # Define a flask app
#     app = Flask(__name__)

#     NAME_OF_FILE = 'model_best' # Name of your exported file
#     PATH_TO_MODELS_DIR = Path('') # by default just use /models in root dir
#     classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
#             'Dermatofibroma', 'Melanocytic nevi', 'Glycoma', 'Vascular lesions']

#     def setup_model_pth(path_to_pth_file, learner_name_to_load, classes):
#         data = ImageDataBunch.single_from_classes(
#             path_to_pth_file, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
#         learn = cnn_learner(data, models.densenet169, model_dir='models')
#         learn.load(learner_name_to_load, device=torch.device('cpu'))
#         return learn

#     learn = setup_model_pth(PATH_TO_MODELS_DIR, NAME_OF_FILE, classes)

#     def encode(img):
#         img = (image2np(img.data) * 255).astype('uint8')
#         pil_img = PILImage.fromarray(img)
#         buff = BytesIO()
#         pil_img.save(buff, format="JPEG")
#         return base64.b64encode(buff.getvalue()).decode("utf-8")
        
#     def model_predict(img):
#         img = open_image(BytesIO(img))
#         pred_class,pred_idx,outputs = learn.predict(img)
#         formatted_outputs = ["{:.1f}%".format(value) for value in [x * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]]
#         pred_probs = sorted(
#                 zip(learn.data.classes, map(str, formatted_outputs)),
#                 key=lambda p: p[1],
#                 reverse=True
#             )
        
#         img_data = encode(img)
#         result = {"class":pred_class, "probs":pred_probs, "image":img_data}
#         return render_template('result.html', result=result)
    

#     @app.route('/', methods=['GET', "POST"])
#     def index():
#         # Main page
#         return render_template('index.html')


#     @app.route('/upload', methods=["POST", "GET"])
#     def upload():
#         if request.method == 'POST':
#             # Get the file from post request
#             img = request.files['file'].read()
#             if img != None:
#             # Make prediction
#                 preds = model_predict(img)
#                 return preds
#         return 'OK'
        
#     @app.route("/classify-url", methods=["POST", "GET"])
#     def classify_url():
#         if request.method == 'POST':
#             url = request.form["url"]
#             if url != None:
#                 response = requests.get(url)
#                 preds = model_predict(response.content)
#                 return preds
#         return 'OK'
        

#     if __name__ == '__main__':
#         port = os.environ.get('PORT', 8008)

#         if "prepare" not in sys.argv:
#             app.run(debug=False, host='0.0.0.0', port=port)
# except:
#     print("hai")

# from tkinter import *
# from tkinter import messagebox
# from keras.models import load_model
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# import os
# import h5py
# import pandas


# model = load_model('f1.h5')
# #with h5py.File(h5file,'r') as fid:
#  #    model = load_model(fid)


# def get_filenames():
#     global path
#     path = r"test"
#     return os.listdir(path)


# def curselect(event):
#     global spath
#     index = t1.curselection()[0]
#     spath = t1.get(index)
#     return(spath)


# def autoroi(img):

#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     thresh = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)[1]
#     thresh = cv2.dilate(thresh, None, iterations=5)

#     contours, hierarchy = cv2.findContours(
#         thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     biggest = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(biggest)
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     roi = img[y:y+h, x:x+w]

#     return roi


# def prediction():

#     img = cv2.imread('test/%s' % (spath))
#     img = autoroi(img)
#     img = cv2.resize(img, (256, 256))
#     img = np.reshape(img, [1, 256, 256, 3])

#     prob = model.predict(img)
#     Class = prob.argmax(axis=-1)

#     return(Class)


# def run():

#     Class = prediction()
#     if (Class == 0):
#         messagebox.showinfo('Prediction', 'You have been diagnosed with Glaucoma')
#     else:
#         messagebox.showinfo('Prediction', 'Congratulations! You are Healthy')


# def run_all():

#     x = os.listdir(path)
#     y = []
#     affected = 0

#     for i in x:
#         img = cv2.imread('test/%s' % (i))
#         img = autoroi(img)
#         img = cv2.resize(img, (256, 256))
#         img = np.reshape(img, [1, 256, 256, 3])

#         prob = model.predict(img)
#         Class = prob.argmax(axis=-1)
#         y.append(Class[0])
#         if Class == 1:
#             affected += 1

#     df = pandas.DataFrame(data=y, index=x, columns=["output"])
#     df.to_csv('output.csv', sep=',')


# def ROI():
#     img = cv2.imread('test/%s' % (spath))
#     roi = autoroi(img)
#     cv2.imshow("Region of Interest", roi)


# def preview():
#     img = cv2.imread('test/%s' % (spath))
#     cv2.imshow('Image', img)


# def graph():

#     total = len(os.listdir(path))
#     affected = pandas.read_csv('output.csv')
#     affected = affected['output'].sum()

#     healthy = total - affected

#     piey = ["Glaucomatous", "Healthy"]
#     piex = [affected, healthy]

#     plt.axis("equal")
#     plt.pie(piex, labels=piey, radius=1.5, autopct='%0.1f%%', explode=[0.2, 0])
#     plt.show()

# # Frontend GUI


# window = Tk()
# window.title("Glaucoma Detection")
# window.geometry('1000x550')
# window.configure(background='grey')

# l1 = Label(window, text="Test Image", font=("Arial", 20), padx=10, bg='grey')
# l1.grid(row=0, column=0)

# b1 = Button(window, text='Run', font=("Arial", 20), command=run)
# b1.grid(row=1, column=3)

# b2 = Button(window, text='Preview', font=("Arial", 20), command=preview)
# b2.grid(row=1, column=2, rowspan=2, padx=10)

# b2 = Button(window, text='ROI', font=("Arial", 20), command=ROI)
# b2.grid(row=2, column=2, rowspan=3, padx=10)

# b3 = Button(window, text='Run all', font=("Arial", 20), command=run_all)
# b3.grid(row=2, column=3)

# b4 = Button(window, text='Graph', font=("Arial", 20), command=graph)
# b4.grid(row=3, column=3)

# t1 = Listbox(window, height=20, width=60, selectmode=SINGLE, font=("Arial", 15), justify=CENTER)
# t1.grid(row=1, column=0, rowspan=3, padx=10)
# for filename in get_filenames():
#     t1.insert(END, filename)
# t1.bind('<<ListboxSelect>>', curselect)

# sb1 = Scrollbar(window)
# sb1.grid(row=1, column=1, rowspan=4)

# t1.configure(yscrollcommand=sb1.set)
# sb1.configure(command=t1.yview)


# window.mainloop()
