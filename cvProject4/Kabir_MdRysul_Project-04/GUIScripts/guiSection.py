# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:51:53 2019

@author: xwc981
"""
from PIL import ImageTk, Image
from tkinter import ttk # this is the extended themed tk
import tkinter as tk
import cv2 as cv2
import numpy as np
import glob
import json 

class VideoSection:
    def __init__(self,parent): 
        
        ttkStyle = ttk.Style()
        ttkStyle.configure('TLabelframe.Label',font='arial 14 bold')
        self.video_section_frame = parent
        self.camera_capture = cv2.VideoCapture(1)
        self.captured_image_bgr=''
        self.captured_image = ''
        
        # initial display
        path = "utrgv2.png"  
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.fixed_image = Image.fromarray(image)
        
        self.lbl1='' 
        self.lbl2=''
        self.lbl3=''
        
        self.imgTk = ImageTk.PhotoImage(self.fixed_image.resize((300, 200), Image.ANTIALIAS))
        self.videoLabel = ttk.Label(self.video_section_frame, image=self.imgTk,anchor=tk.CENTER, justify=tk.RIGHT)        
        self.videoLabel.image = self.imgTk
        
        # fields related to I/O of key images
        self.file_name = tk.StringVar()   
        self.save_captured_file_button=''
        
        # fields related to I/O for key information
        self.key_owner = tk.StringVar()  
        self.key_address = tk.StringVar() 

    def start_stream(self):
        self.camera_capture.release()
        self.camera_capture = cv2.VideoCapture(1) 
        self.show_frame()
        
    def stop_stream(self):
        self.camera_capture.release()
        cv2.destroyAllWindows()
    
    def show_frame(self):
        _, frame = self.camera_capture.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        image = Image.fromarray(cv2image) 
        
        # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
        self.imgTk = ImageTk.PhotoImage(image.resize((300, 200), Image.ANTIALIAS))
        self.videoLabel.configure(image=self.imgTk)
        self.videoLabel.after(10,self.show_frame)
    
    def shut_camera(self):
        self.camera_capture.release()
        cv2.destroyAllWindows()
    
    def capture_frame(self):
        _, frame = self.camera_capture.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        self.captured_image_bgr = frame
        self.captured_image = Image.fromarray(cv2image)
        
        self.captured_imgTk = ImageTk.PhotoImage(self.captured_image.resize((300, 200), Image.ANTIALIAS))
        self.captured_imgTk_Label.configure(image=self.captured_imgTk)
 
    
    def save_captured_file(self):
        if self.file_name.get():
            cv2.imwrite('images/'+self.file_name.get(), self.captured_image_bgr)
        else:
            tk.messagebox.showwarning(message='no file name was found')
   
    
    def save_keyInfo_file(self):
        return
                                      
    def video_section(self,labelFrameText):                
        self.lbl1 = ttk.Label(self.video_section_frame, text="CSCI 6368: Computer Vision, Object Matching by ORB", foreground="Navy",font=("Helvetica", 16))
        self.lbl2 = ttk.Label(self.video_section_frame, text="Video Stream Panel ", foreground="Navy",font=("Helvetica", 12))
#        self.lbl3 = ttk.Label(self.video_section_frame, text="", foreground="Navy")
        
        self.lbl1.grid(column=0, row=1, columnspan=2)
        self.lbl2.grid(column=0, row=2, columnspan=2)
#        self.lbl3.grid(column=1, row=3)
        
        # video frames
        self.videoLabel.grid(column=0, row=3,sticky='EW')

        # Video Control Buttons
        self.captured_frame = tk.Frame(self.video_section_frame)
        self.captured_imgTk = ImageTk.PhotoImage(self.fixed_image.resize((300, 200), Image.ANTIALIAS))
        self.captured_imgTk_Label = ttk.Label(self.captured_frame, image=self.imgTk,anchor=tk.CENTER, justify=tk.RIGHT)        
        self.captured_imgTk_Label.image = self.captured_imgTk 
        self.capture_button=ttk.Button(self.captured_frame, text="Capture Frame",command=self.capture_frame)
    
        self.captured_imgTk_Label.grid(row=0,column=0)
        self.capture_button.grid(row=1,column=0)
        
        self.captured_frame.grid(column=1,row=3,rowspan=4,sticky="nsew")
        
        # I/O Operations
        # 
        io_operations = ttk.LabelFrame(self.video_section_frame, text=' I/O Operations ')
        io_operations.grid(column=0, row=8,rowspan=4,columnspan=6, padx=10, pady=5,sticky='EW')
        
        # 
        fname_label = ttk.Label(io_operations, text="Enter a file name for captured image (with.jpg ext):")
        fname_label.grid(column=0, row=0, sticky='W')    
        
        fname_entered = ttk.Entry(io_operations, width=50, textvariable=self.file_name)
        fname_entered.grid(column=0, row=1, sticky='W') 
        
        # Adding a save Button
        self.save_captured_file_button  = ttk.Button(io_operations, text="Save Me!", command=self.save_captured_file)   
        self.save_captured_file_button.grid(column=2, row=1) 
        # add other fields as needed...
    
        
        return self.video_section_frame
    
    
class TrainingSection:
    def __init__(self,parent): 
        
        ttkStyle = ttk.Style()
        ttkStyle.configure('TLabelframe.Label',font='arial 14 bold')
        self.training_section_frame = parent
        #self.camera_capture = cv2.VideoCapture(1)
        #self.captured_image_bgr=''
        #self.captured_image = ''
        
        # initial display
        path = "utrgv2.png"  
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.fixed_image = Image.fromarray(image)
        
        self.lbl1='' 
        self.lbl2=''
        self.lbl3=''
        
        self.imgTk = ImageTk.PhotoImage(self.fixed_image.resize((300, 200), Image.ANTIALIAS))
        self.videoLabel = ttk.Label(self.training_section_frame, image=self.imgTk,anchor=tk.CENTER, justify=tk.RIGHT)        
        self.videoLabel.image = self.imgTk
        
        # fields related to I/O of key images
        self.pathFilename = tk.StringVar()   
        self.objectName = tk.StringVar() 
        #self.save_captured_file_button=''
        
        # fields related to I/O for key information
        #self.key_owner = tk.StringVar()  
        #self.key_address = tk.StringVar() 

    def start_stream(self):
        self.camera_capture.release()
        self.camera_capture = cv2.VideoCapture(1) 
        self.show_frame()
        
    def stop_stream(self):
        self.camera_capture.release()
        cv2.destroyAllWindows()
    
    def show_frame(self):
        img = cv2.imread('images/'+self.pathFilename.get())
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        image = Image.fromarray(cv2image) 
        
        # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
        self.imgTk = ImageTk.PhotoImage(image.resize((300, 300), Image.ANTIALIAS))
        self.videoLabel.configure(image=self.imgTk)
        #self.videoLabel.after(10,self.show_frame)
    
    def shut_camera(self):
        self.camera_capture.release()
        cv2.destroyAllWindows()
            
           
    def object_training(self):
        img = cv2.imread('images/'+self.pathFilename.get())
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.imgGrayResized = cv2.resize(imgGray, (800,600), interpolation = cv2.INTER_AREA)
        
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # compute the descriptors and keypoints with ORB
        kp, des = orb.detectAndCompute(self.imgGrayResized,None)
        #print(des.shape)
        #print(des)
        # saving the descriptors 
        np.savetxt('orbs/'+self.objectName.get()+'.orb', des, delimiter=',', fmt='%d')
        
        # adding the object name in a json file
        # load the json file and append the new object name
        with open("objects.json", 'r') as f:
            objects = json.load(f)
            objects.append(self.objectName.get())
        # writing the new object name in the object json file
        with open("objects.json", 'w') as f:
           json.dump(objects, f, indent=2)
        
        
        
        img2 = cv2.drawKeypoints(self.imgGrayResized, kp, None, color=(0,255,0), flags=0)
        
        cv2image = cv2.cvtColor(img2, cv2.COLOR_BGR2RGBA)
        self.selected_image_bgr = img2
        self.selected_image = Image.fromarray(cv2image)
        
        self.selected_imgTk = ImageTk.PhotoImage(self.selected_image.resize((300, 300), Image.ANTIALIAS))
        self.selected_imgTk_Label.configure(image=self.selected_imgTk)
           
    
    def training_section(self,labelFrameText):                
        self.lbl1 = ttk.Label(self.training_section_frame, text="UTRGV Object Matching System", foreground="Navy",font=("Helvetica", 16))
        #self.lbl2 = ttk.Label(self.training_section_frame, text="Video Stream Panel ", foreground="Navy",font=("Helvetica", 12))
#        self.lbl3 = ttk.Label(self.training_section_frame, text="", foreground="Navy")
        
        self.lbl1.grid(column=0, row=1, columnspan=2)
        #self.lbl2.grid(column=0, row=2, columnspan=2)
#        self.lbl3.grid(column=1, row=3)
        
        # video frames
        self.videoLabel.grid(column=0, row=3,sticky='EW')

        # Video Control Buttons
        self.selected_frame = tk.Frame(self.training_section_frame)
        self.selected_imgTk = ImageTk.PhotoImage(self.fixed_image.resize((300, 200), Image.ANTIALIAS))
        self.selected_imgTk_Label = ttk.Label(self.selected_frame, image=self.imgTk,anchor=tk.CENTER, justify=tk.RIGHT)        
        self.selected_imgTk_Label.image = self.selected_imgTk 
        
        
    
        self.selected_imgTk_Label.grid(row=0,column=0)
        
        
        self.selected_frame.grid(column=1,row=3,rowspan=4,sticky="nsew")
        
        
        
        # object training i.e. creating descriptors and showing them off on the image
        
        
        # I/O Operations
        # 
        io_operations = ttk.LabelFrame(self.training_section_frame, text=' I/O Operations ')
        io_operations.grid(column=0, row=8,rowspan=4,columnspan=6, padx=10, pady=5,sticky='EW')
        
        # 
        pname_label = ttk.Label(io_operations, text="Enter a file name for the object image (with.jpg ext):")
        pname_label.grid(column=0, row=0, sticky='W')    
        
       
        
        pname_entered = ttk.Entry(io_operations, width=50, textvariable=self.pathFilename)
        pname_entered.grid(column=0, row=2, sticky='W') 
        
        self.show_button=ttk.Button(io_operations, text="Show Image",command=self.show_frame)
        # add other fields as needed...
        self.show_button.grid(row=2,column=1, padx=2)
        
        oname_label = ttk.Label(io_operations, text="Enter the name of the object")
        oname_label.grid(column=0, row=3, sticky='W')    
        
        oname_entered = ttk.Entry(io_operations, width=50, textvariable=self.objectName)
        oname_entered.grid(column=0, row=4, sticky='W') 
        self.selected_button=ttk.Button(io_operations, text="Train",command=self.object_training)
        self.selected_button.grid(row=4,column=1)
        
        return self.training_section_frame



# show the match add later after completing the writing

class ClassificationSection:
    def __init__(self,parent): 
        
        ttkStyle = ttk.Style()
        ttkStyle.configure('TLabelframe.Label',font='arial 14 bold')
        self.classification_section_frame = parent
        #self.camera_capture = cv2.VideoCapture(1)
        #self.captured_image_bgr=''
        #self.captured_image = ''
        
        # initial display
        path = "utrgv2.png"  
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.fixed_image = Image.fromarray(image)
        self.matchingMetric = cv2.NORM_HAMMING
        self.lbl1='' 
        self.lbl2=''
        self.lbl3=''
        
        self.imgTk = ImageTk.PhotoImage(self.fixed_image.resize((300, 200), Image.ANTIALIAS))
        self.videoLabel = ttk.Label(self.classification_section_frame, image=self.imgTk,anchor=tk.CENTER, justify=tk.RIGHT)        
        self.videoLabel.image = self.imgTk
        self.objName = 'No Object Found'
        # fields related to I/O of key images
        self.pathFilename = tk.StringVar()   
        self.objectName = tk.StringVar() 
        #self.save_captured_file_button=''
        
        # fields related to I/O for key information
        #self.key_owner = tk.StringVar()  
        #self.key_address = tk.StringVar() 

    def start_stream(self):
        self.camera_capture.release()
        self.camera_capture = cv2.VideoCapture(1) 
        self.show_frame()
        
    def stop_stream(self):
        self.camera_capture.release()
        cv2.destroyAllWindows()
    
    def show_frame(self):
        img = cv2.imread('images/'+self.pathFilename.get())
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        image = Image.fromarray(cv2image) 
        
        # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
        self.imgTk = ImageTk.PhotoImage(image.resize((300, 300), Image.ANTIALIAS))
        self.videoLabel.configure(image=self.imgTk)
        #self.videoLabel.after(10,self.show_frame)
    
    def shut_camera(self):
        self.camera_capture.release()
        cv2.destroyAllWindows()
    
    
    def loadFiles(self, path, mode = 'orb'):
        '''
        loads all the files within that path (for getting all the orb files and the query images)
        returns a list with the orb file descriptors or image arrays
        '''
        fileContents = []
        orbNames = []
        for fileName in glob.glob(path):
            if (mode == 'image'):
                img = cv2.imread(fileName)
                imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                fileContent = cv2.resize(imgGray, (800,600), interpolation = cv2.INTER_AREA)
            elif (mode == 'orb'):
                fileContent = np.loadtxt(fileName, delimiter=',', dtype = np.uint8) # loading should be in np.unit8
            
            f = fileName.split('.')[0]
            orbNames.append(f.split('/')[1])
            fileContents.append(fileContent)
        
        return fileContents, orbNames
           
    def object_classification(self):
        img = cv2.imread('images/'+self.pathFilename.get())
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.imgGrayResized = cv2.resize(imgGray, (800,600), interpolation = cv2.INTER_AREA)
        
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # compute the descriptors and keypoints with ORB
        kp, des = orb.detectAndCompute(self.imgGrayResized,None)
                # get the object names with orb feature files
        #with open("objects.json", 'r') as f:
        #    featureObjects = json.load(f)
    
        
        # loading all the orb descriptor files
        orbFeatures, featureObjects = self.loadFiles(path = "orbs/*")
        # check the sequentiality of the 
        orbMatches = []
        
        for desCheck in orbFeatures:
            # create BFMatcher object
            bf = cv2.BFMatcher(self.matchingMetric, crossCheck=True)
            #print(des1.shape)
            #print(desCheck.shape)
            # Match descriptors.
            matches = bf.match(des,desCheck)
        
            #print(len(matches))
        
            matchDistances=[]
            for match in matches:
                matchDistances.append(match.distance)
        
            #print(min(matchDistances))
            orbMatches.append(min(matchDistances))
        
    
        
        self.objName = "There is a "+"'"+featureObjects[orbMatches.index(min(orbMatches))]+"'"+" in the scene."
        self.oname_label.config(text = self.objName)
        
        
        
        '''
        img2 = cv2.drawKeypoints(self.imgGrayResized, kp, None, color=(0,255,0), flags=0)
        
        cv2image = cv2.cvtColor(img2, cv2.COLOR_BGR2RGBA)
        self.selected_image_bgr = img2
        self.selected_image = Image.fromarray(cv2image)
        
        self.selected_imgTk = ImageTk.PhotoImage(self.selected_image.resize((300, 300), Image.ANTIALIAS))
        self.selected_imgTk_Label.configure(image=self.selected_imgTk)
        '''  
    
    def classification_section(self,labelFrameText):                
        self.lbl1 = ttk.Label(self.classification_section_frame, text="UTRGV Object Matching System", foreground="Navy",font=("Helvetica", 16))
        #self.lbl2 = ttk.Label(self.classification_section_frame, text="Video Stream Panel ", foreground="Navy",font=("Helvetica", 12))
#        self.lbl3 = ttk.Label(self.video_section_frame, text="", foreground="Navy")
        
        self.lbl1.grid(column=0, row=1, columnspan=2)
        #self.lbl2.grid(column=0, row=2, columnspan=2)
#        self.lbl3.grid(column=1, row=3)
        
        # video frames
        self.videoLabel.grid(column=0, row=3,sticky='EW')

        # Video Control Buttons
        self.selected_frame = tk.Frame(self.classification_section_frame)
        self.selected_imgTk = ImageTk.PhotoImage(self.fixed_image.resize((300, 200), Image.ANTIALIAS))
        self.selected_imgTk_Label = ttk.Label(self.selected_frame, image=self.imgTk,anchor=tk.CENTER, justify=tk.RIGHT)        
        self.selected_imgTk_Label.image = self.selected_imgTk 
        
        
    
        self.selected_imgTk_Label.grid(row=0,column=0)
        
        
        
        # I/O Operations
        # 
        self.io_operations = ttk.LabelFrame(self.classification_section_frame, text=' I/O Operations ')
        self.io_operations.grid(column=0, row=8,rowspan=4,columnspan=6, padx=10, pady=5,sticky='EW')
        
        # 
        pname_label = ttk.Label(self.io_operations, text="Enter a file name for the object image (with.jpg ext):")
        pname_label.grid(column=0, row=0, sticky='W')    
        
       
        
        pname_entered = ttk.Entry(self.io_operations, width=50, textvariable=self.pathFilename)
        pname_entered.grid(column=0, row=2, sticky='W') 
        
        self.show_button=ttk.Button(self.io_operations, text="Show Image",command=self.show_frame)
        # add other fields as needed...
        self.show_button.grid(row=2,column=1, padx=2)
        
        
        self.selected_button=ttk.Button(self.io_operations, text='Classify',command=self.object_classification)
        self.selected_button.grid(row=3,column=0,padx=10)
        
        self.name_label = ttk.Label(self.io_operations, text='Detected Object: ' )
        self.name_label.grid(column=0, row=4, sticky='W')    
        
        self.oname_label = ttk.Label(self.io_operations, text=self.objName, font=("Arial", 18) )
        self.oname_label.grid(column=0, row=5, sticky='W')    
        
        
        return self.classification_section_frame
