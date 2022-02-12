import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
#img = cv2.imread('1.jpg', )
cap = cv2.VideoCapture('train.mp4')
save={}
def all():
    global save
    cadr = 1
    while (1):
        try:
            _, frame = cap.read()
            #print(f'Номер кадра{cadr}')
            new_image = np.zeros(frame.shape, frame.dtype)
            alpha = 0.84  # керування контрастністю
            beta = 2.7  # регулювання яскравості
            new_image[:frame.shape[0], :frame.shape[1],
                :frame.shape[2]] = np.clip(alpha * np.array(frame) + beta, 0, 240)
            # Ініціалізація значень
            ret1, th1 = cv2.threshold(new_image, 127, 255, cv2.THRESH_BINARY)  # Проста фільтрация
            colorl = np.array([248, 248, 248])
            coloru = np.array([255, 255, 255])
            mask = cv2.inRange(th1, colorl, coloru)

            contours, _ = cv2.findContours(mask, 1, 2)
            #print(contours)
            cont=[]
            g=[]
            #print(cv2.contourArea(contours[1],oriented = False))
            for i in range(len(contours)):
                print(cv2.contourArea(contours[i], oriented=False))
                if cv2.contourArea(contours[i], oriented=False)<12200:
                    cont.append(contours[i])
                    g.append(cv2.contourArea(contours[i], oriented=False))
            save.update({cadr: g})

            #print(cs)
            cv2.drawContours(th1, cont, -1, (0,255,0), 3)
            cv2.namedWindow("window", cv2.WND_PROP_AUTOSIZE)
            cv2.setWindowProperty("window", cv2.WND_PROP_AUTOSIZE,
                                 cv2.WINDOW_FULLSCREEN)
            cv2.imshow('window', th1)
            cadr+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            break
    cv2.destroyAllWindows()

all()


def rozra(a,tst):
    lst = save[a]
    g=[]
    lst.sort()
    k=0
    l=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(len(lst)):
        if lst[i] in range(30,251):
            k+=1
    l[0]=k
    for i in range(len(lst)):
        for j in range(1,len(l)):
            if lst[i] in range(250*j,250*(j+1)):
                l[j]+=1
                break
    print(l)
    groups = [f"{250*i}" for i in range(len(l))]
    counts = l
    plt.bar(groups, counts)
    plt.show()
    k=sum([(l[i]-tst[i])**2 for i in range(len(l))])/len(l)
    print(f'MSE = {k}')
tst54=[26., 55., 44., 37., 31., 20., 24., 12.,  6., 10.,  4.,  1.,  1.,
         0.,  1.,  0.,  0.,  1.,  0.,  0.]
tst101=[25., 63., 54., 41., 37., 15., 14., 12.,  3.,  5.,  4.,  2.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.]
tst256=[36., 64., 70., 46., 49., 24., 20., 19.,  9.,  5.,  5.,  3.,  2.,
         0.,  1.,  0.,  0.,  0.,  0.,  0.]

rozra(54,tst54)
rozra(101,tst101)
rozra(256,tst256)
