# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:49:57 2019
Video Capture GUI
@author: xwc981
"""
# =============================================================================
import tkinter as tk
from tkinter import ttk # this is the extended themed tk
import tkinter.messagebox as msb
import sys
import guiSection as gui

class MainGUI: 
    
    def on_closing(self):
        if msb.askokcancel("Quit", "Do you want to quit?"):
            self.video_info.shut_camera()
            self.win.destroy()

    def quit_form(self):
        self.win.quit()
        self.win.destroy()
        sys.exit()
        
#    def create_training_viewer(self):
#        self.tabControl.select(self.tab2)
#        self.train_view.pack(side="top", expand=1, fill="both")
#        self.tabControl.select(self.tab2)        
    
    def __init__(self):
        self.win = tk.Tk()   
        self.win.geometry("800x600") # size of windows
        self.win.resizable(1, 1)
        self.win.title("CSCI 6368: Computer Vision Project 4")
        self.win.iconbitmap(r'logo.ico')
        self.win.geometry("+50+30") # distance from left and top

        # =============================================================================
        # Tab Control
        # =============================================================================
        self.tabControl = ttk.Notebook(self.win)          # Create Tab Control
        
        self.tab1 = ttk.Frame(self.tabControl)            # Create a tab 
        self.tab1 = ttk.Frame(self.win)                    # Create a tab 
        self.tabControl.insert('end',self.tab1, text='Video Stream')      # Add the tab
        
        self.tab2 = ttk.Frame(self.tabControl)                         # Add a second tab
        self.tabControl.insert('end',self.tab2, text='Training Stage')                
        self.tabControl.pack(expand=1, fill="both") 
        
        self.tab3 = ttk.Frame(self.tabControl)                      
        self.tabControl.insert('end',self.tab3, text='Classification Stage')                
        self.tabControl.pack(expand=1, fill="both") 
        
        
        self.tab1.grid_rowconfigure(0, weight=1)
        self.tab1.grid_columnconfigure(0, weight=1)
        
        self.tab2.grid_rowconfigure(0, weight=1)
        self.tab2.grid_columnconfigure(0, weight=1)
        
        self.tab3.grid_rowconfigure(0, weight=1)
        self.tab3.grid_columnconfigure(0, weight=1)
        
        #     
        self.video_frame = tk.Frame(self.tab1, padx=2, pady=1)
        
        # Insert GUI components
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(1, weight=0)
        self.video_frame.grid_columnconfigure(2, weight=1)
        
        self.video_info = gui.VideoSection(self.video_frame)
        self.video_information = self.video_info.video_section('Key Streaming')
        self.video_information.grid(column=0, row=0,sticky=tk.EW+tk.NS)
        self.video_info.show_frame()
        
            
        self.training_frame = tk.Frame(self.tab2, padx=2, pady=1)
        
        # Insert GUI components
        self.training_frame.grid_columnconfigure(0, weight=1)
        self.training_frame.grid_columnconfigure(1, weight=0)
        self.training_frame.grid_columnconfigure(2, weight=1)
        
        #ttk.Label(self.tab2, text="This is Tab 2").grid(column=0, row=0, padx=10, pady=10)
        self.training_info = gui.TrainingSection(self.training_frame)
        self.training_information = self.training_info.training_section('Training Streaming')
        self.training_information.grid(column=0, row=0,sticky=tk.EW+tk.NS)
        
        
        self.classification_frame = tk.Frame(self.tab3, padx=2, pady=1)
        
        # Insert GUI components
        self.classification_frame.grid_columnconfigure(0, weight=1)
        self.classification_frame.grid_columnconfigure(1, weight=0)
        self.classification_frame.grid_columnconfigure(2, weight=1)
        
        #ttk.Label(self.tab2, text="This is Tab 2").grid(column=0, row=0, padx=10, pady=10)
        self.classification_info = gui.ClassificationSection(self.classification_frame)
        self.classification_information = self.classification_info.classification_section('Classification Streaming')
        self.classification_information.grid(column=0, row=0,sticky=tk.EW+tk.NS)
        
        
        self.win.protocol("WM_DELETE_WINDOW", self.on_closing)   
        self.win.mainloop() 
        
        self.tabControl.select(self.tab1)

if __name__ == "__main__":
    MainGUI() 
else:
    print("Code is being imported into another module") 
        
