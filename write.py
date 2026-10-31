from PIL import Image as PILImage
from PIL import ImageDraw as PILDraw
from PIL import ImageTk
from PIL import ImageOps
from tkinter import *
import numpy as np

def preprocess_image(img):
    box_min = img.getbbox()
    if box_min is None:
        return None
    img = img.crop(box_min)
    w,h = img.size
    side= max(w,h)
    side += 2 * int(side * 0.25)
    square_img = PILImage.new("L",(side,side),0)
    square_img.paste(img,((side - w) // 2, (side - h) // 2))
    res = square_img.resize((8,8),PILImage.Resampling.BOX)
    return ImageOps.autocontrast(res)

def convert_img(img):
    a = np.asarray(img, dtype=float)
    a = a / a.max() # Contraste
    return a.flatten() # Vecteur colonne

class DrawingApp:
    def __init__(self,window,network=None):
        self.network = network
        
        self.window = window
        self.window.title("Reconnaissance de Chiffres Manuscripts")
        self.window.geometry("800x600")
        self.window.resizable(False, False)
        
        frame_left = Frame(window, borderwidth=2, relief=GROOVE)
        frame_left.pack(side=LEFT, padx=30, pady=30)
        
        self.canvas = Canvas(frame_left,width=280,height=280,bg="black")
        self.canvas.pack()
        erase_button=Button(frame_left, text="Effacer", command=self.erase)
        erase_button.pack()
        
        frame_right = Frame(window, borderwidth=2, relief=GROOVE)
        frame_right.pack(side=RIGHT, padx=30, pady=30)
        
        self.canvas_input = Canvas(frame_right, width=160, height=160, bg="black", highlightthickness=0)
        self.canvas_input.pack()
        
        self.pen_color = "white"
        self.drawing = False
        self.last_x,self.last_y = None,None 
        
        self.img = PILImage.new("L",(280,280),0)
        self.drawn_input = PILDraw.Draw(self.img)
        self.shown_img = None
        
        self.canvas.bind("<Button-1>", self.draw)
        self.canvas.bind("<B1-Motion>", self.show)
        self.canvas.bind("<ButtonRelease-1>", self.stop)

    
    def draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
        
    def show(self,event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill=self.pen_color, width=6, capstyle=ROUND)
            self.drawn_input.line([self.last_x,self.last_y,x,y],fill=255,width=6)
            self.last_x, self.last_y = x, y
    
    def stop(self,event):
        self.drawing = False
        
        img = preprocess_image(self.img)
        big_img = img.resize((160, 160), PILImage.Resampling.NEAREST)
        self.shown_img = ImageTk.PhotoImage(big_img)
        self.canvas_input.create_image(0,0,anchor=NW,image=self.shown_img)

        print(self.predict(convert_img(img)))

    def erase(self):
        self.canvas.delete("all")
        self.canvas_input.delete("all")
        self.drawn_input.rectangle([0, 0, 280,280], fill=0)
        
    def predict(self,inputs):
        outputs = self.network.evaluate(self.network.forward_pass(inputs))
        digit_predicted = int(outputs)
        return digit_predicted