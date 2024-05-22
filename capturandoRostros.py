import cv2
import os
import imutils
from tkinter import *
from tkinter import ttk

# Directorio del archivo actual
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

# Directorio para Guardar las caras
dataPath = current_directory+'/Caras'

# Funcion para obtener el nombre de la persona que se va a capturar
def obtener_nombre_persona():
    root = Tk()
    root.geometry('300x150')
    frm = ttk.Frame(root, padding=10)
    frm.grid()
    ttk.Label(frm, text="Ingresa el nombre de la persona").grid(column=0, row=0)
    name = StringVar()
    ttk.Entry(frm, justify='center', textvariable=name, background='white', font='blue').grid(column=0, row=1)
    ttk.Button(frm, text="Aceptar", command=root.destroy).grid(column=0, row=2)
    root.mainloop()
    return name

def crear_carpeta(name):
    personPath = dataPath + '/' + name
    if not os.path.exists(personPath):
        print('Carpeta creada: ', personPath)
        os.makedirs(personPath)
    return personPath

def abrir_camara():
    print("Abriendo camara")
    cap = cv2.VideoCapture(0) #Nombre del video de entrenamiento
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    return cap, faceClassif

def capturar(cap, faceClassif, personPath):
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False: break

        frame = imutils.resize(frame, width=640, height=200)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, 200), (x+w, y+h), (0, 255, 0), 2)
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
            count += 1
        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        if k == 27 or count >= 100:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Configuracion
    name = obtener_nombre_persona()
    personPath = crear_carpeta(name.get())
    # Captura
    cap, faceClassif = abrir_camara()
    capturar(cap, faceClassif, personPath)

    print("Listo!")
