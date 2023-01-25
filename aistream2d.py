import cv2
import pygame
import numpy as np
import dlib
from math import hypot
count=0
cap = cv2.VideoCapture(0)


pygame.init()
screen=pygame.display.set_mode((1280,720))
pygame.display.set_caption("Ai Streamer")
bg=pygame.image.load("bg.png")
eo=pygame.image.load('eyeopen.png')
ec=pygame.image.load('eyeclose.png')
om=pygame.image.load('mouthopen.png')
cm=pygame.image.load('mouthclose.png')


detector = dlib.get_frontal_face_detector()



predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def background():
    screen.blit(bg,(0,0))
def open():
    screen.blit(eo,(500,300))
    
def close():
    screen.blit(ec,(500,320))
def closemouth():
    screen.blit(cm,(520,380))
def openmouth():
    screen.blit(om,(520,380))

running=True




def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio


def get_mouth_ratio(mouth_points,facial_landmarks):#60,50,52,64,56,58
    left_point = (facial_landmarks.part(mouth_points[0]).x, facial_landmarks.part(mouth_points[0]).y)
    right_point = (facial_landmarks.part(mouth_points[3]).x, facial_landmarks.part(mouth_points[3]).y)
    center_top = midpoint(facial_landmarks.part(mouth_points[1]), facial_landmarks.part(mouth_points[2]))
    center_bottom = midpoint(facial_landmarks.part(mouth_points[5]), facial_landmarks.part(mouth_points[4]))
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio


def headtilt(facial_landmarks):
    top=(facial_landmarks.part(27).x, facial_landmarks.part(27).y)
    bottom=(facial_landmarks.part(8).x, facial_landmarks.part(8).y)
    hor_line = cv2.line(frame, top, bottom, (0, 255, 0), 2)



mou_ratio=0
blinking_ratio=0

while running:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        mou_ratio=get_mouth_ratio([60,50,52,64,56,58],landmarks)
        headtilt(landmarks)

       
        
    for event in pygame.event.get():
        if event.type==pygame.QUIT :
            running=False
    background()
    
    if blinking_ratio > 5.7:
            close()
    else:
            open()
    if mou_ratio <2.3:
        openmouth()
    else:
        closemouth()
    

    pygame.display.update()

    

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()