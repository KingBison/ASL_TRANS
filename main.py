
from cv2 import ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pygame as pg
import pickle
import torch
import time



from util import Button


#colors
black = (0,0,0)


#imgs

practiceButtonImg = pg.image.load('practicebutton.png')
trialsButtonImg = pg.image.load('trialsbutton.png')
timedButtonImg = pg.image.load('timedbutton.png')
backButtonImg = pg.image.load('backbutton.png')

class_names = { 
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'H', 8: 'I', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',  
    22: 'W',23: 'X', 24: 'Y'
}



cam = cv2.VideoCapture(0)
framex, framey = (0,0)
status, frame = cam.read()

if status:
    framex = int(frame.shape[1]/2)
    framey = int(frame.shape[0]/2)
else:
    print('Camera Not Found')
    quit()

maxval = np.amax(frame)
scalar = int(255/maxval)
prep = frame*scalar

pg.init()


buttons = []
buttons.append(Button(practiceButtonImg,[framey+10,int(.5*framex-150),framey+90,int(.5*framex+150)],'practice'))
buttons.append(Button(trialsButtonImg,[framey+110,int(.5*framex-150),framey+190,int(.5*framex+150)],'trials'))
buttons.append(Button(timedButtonImg,[framey+210,int(.5*framex-150),framey+290,int(.5*framex+150)],'timed'))
backbtn = Button(backButtonImg,[framey+210,int(.5*framex-150),framey+290,int(.5*framex+150)],'home')

with open('model.model','rb') as f:

    model = pickle.load(f)



screen = pg.display.set_mode([framex,framey+300])

running = True
gamemode = 'home' 

#trials vars
correctGuesses = 0
rand = np.random.randint(0,24)
while rand == 9:
    rand = np.random.randint(0,24)
currentTrial = class_names[rand]
timestart = False
trippie = False


while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.MOUSEBUTTONUP:
            mouse_loc = pg.mouse.get_pos()
            if gamemode == 'home':
                for b in buttons:
                    if b.hit(mouse_loc):
                        gamemode = b.tag
            else:
                if backbtn.hit(mouse_loc):
                    gamemode = 'home'
                    correctGuesses = 0
                    rand = np.random.randint(0,24)
                    while rand == 9:
                        rand = np.random.randint(0,24)
                    currentTrial = class_names[rand]
                    timestart = False
                    timer = 0

        if event.type == pg.KEYUP:
            if event.key == pg.K_t:
                trippie = not trippie
            if event.key == pg.K_s:
                rand = np.random.randint(0,24)
                while rand == 9:
                    rand = np.random.randint(0,24)
                currentTrial = class_names[rand]

        

    screen.fill(black)

    status, frame = cam.read()

    if status:
        framex = int(frame.shape[1]/2)
        framey = int(frame.shape[0]/2)
    else:
        print('Camera Not Found')
        quit()

    prep = frame
    img = cv2.resize(prep,(framex,framey))
    if trippie:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.rotate(img,rotateCode=ROTATE_90_COUNTERCLOCKWISE)
    img = pg.surfarray.make_surface(img)

    screen.blit(img,(0,0))

    #translation prep
    mid = int(framex/2)
    left_bound = mid-int(.5*framey)
    right_bound = mid+int(.5*framey)

    tr_prep = cv2.resize(frame,(framex,framey))
    tr_prep = cv2.cvtColor(tr_prep,cv2.COLOR_BGR2GRAY)
    tr_prep = tr_prep[::,left_bound:right_bound]
    tr_prep = cv2.resize(tr_prep,(28,28))
    tr_prep = tr_prep.reshape((1,1,28,28))  
    guess = model(torch.tensor(tr_prep, dtype = torch.float))
    guess = np.argmax(guess.detach().numpy())
    guess = class_names[guess]

    
    
    if gamemode == 'home':
        for b in buttons:
            screen.blit(b.img, (b.hitbox[1],b.hitbox[0]))
    else:
        screen.blit(backbtn.img, (backbtn.hitbox[1],backbtn.hitbox[0]))
        pg.draw.line(screen,black,(left_bound,0),(left_bound,framey),3)
        pg.draw.line(screen,black,(right_bound,0),(right_bound,framey),3)
        
    if gamemode == 'practice':
        font = pg.font.Font('freesansbold.ttf', 32)
        text = font.render(guess, True, (255,255,255))
        screen.blit(text,(framex/2,framey+100))
        
    if gamemode == 'trials':
        
        if guess == currentTrial:
            correctGuesses += 1
            rand = np.random.randint(0,24)
            while rand == 9:
                rand = np.random.randint(0,24)
            currentTrial = class_names[rand]

        font = pg.font.Font('freesansbold.ttf', 32)
        text = font.render('GOAL: ' + currentTrial, True, (255,255,255))
        screen.blit(text,(50,framey+100))
        t0 = font.render(guess,True,(255,255,255))
        screen.blit(t0,(framex/2+50,framey+100))
        t2 = font.render(str(correctGuesses),True,(255,255,255))
        screen.blit(t2,(framex-50,framey+100))

    if gamemode == 'timed':
        if timestart == False:
            timestart = True
            timer = time.time() + 60

        if guess == currentTrial:
            correctGuesses += 1
            rand = np.random.randint(0,24)
            while rand == 9:
                rand = np.random.randint(0,24)
            currentTrial = class_names[rand]

        font = pg.font.Font('freesansbold.ttf', 32)
        text = font.render('GOAL: ' + currentTrial, True, (255,255,255))
        screen.blit(text,(50,framey+100))
        t0 = font.render(guess,True,(255,255,255))
        screen.blit(t0,(framex/2+50,framey+100))
        t2 = font.render(str(correctGuesses),True,(255,255,255))
        screen.blit(t2,(framex-50,framey+100))
        t4 = font.render("Time: " + str(int(timer-time.time())),True,(255,255,255))
        screen.blit(t4,(50,framey+50))

        if time.time() > timer:
            gamemode = 'home'
            correctGuesses = 0
            rand = np.random.randint(0,24)
            while rand == 9:
                rand = np.random.randint(0,24)
            currentTrial = class_names[rand]
            timestart = False
            timer = 0




    



    pg.display.flip()



pg.quit()

 





