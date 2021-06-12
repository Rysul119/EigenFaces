#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:29:10 2021

@author: rysul
"""
# tkinter scratch

# uploading an image
# capturing an image
# uploaded/captured image to train/classification window
# if in train window show the key points
# if in classification window say what is the object in the scene


import tkinter as tk

root = tk.Tk()

margin = 0.23

entry = tk.Entry(root)

entry.pack()

def profit_calculator():
    profit = margin * int(entry.get())
    print(profit)

button_calc = tk.Button(root, text="Calculate", command=profit_calculator)
button_calc.pack()

root.mainloop()