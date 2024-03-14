import cv2
import pandas as pd
import sounddevice as sd  # for the record
from scipy.io.wavfile import write  # to save the file
import numpy as np
import soundfile  # for converting the audio format
import speech_recognition as sr  # for speech to text
import time
import pyttsx3  # for text-to-speech

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# load the model
net = cv2.dnn.readNet("C:/Users/jalaj/OneDrive/Desktop/VoiceBlind/yolov4-tiny.weights",
                      "C:/Users/jalaj/OneDrive/Desktop/VoiceBlind/yolov4-tiny.cfg")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

# load the classes names and store in a list
classesNames = []
df = pd.read_csv("classes.txt", header=None, names=["ClassName"])
for index, row in df.iterrows():
    ClassName = df.iloc[index]['ClassName']
    classesNames.append(ClassName)

#print(classesNames)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# button dims
x1 = 20
y1 = 20
x2 = 570
y2 = 90

fs = 44100  # audio rate
seconds = 3  # duration
audioFileName = "C:/Users/jalaj/OneDrive/Desktop/VoiceBlind/output.wav"

ButtonFlag = False
LookForThisClassName = ""
detected_flag = False  # Flag to track if an object has been detected
detected_time = 0  # Time when an object is detected

# Count of detected objects
detected_count = 0

# capture the mouse, get the left click and record audio
def recordAudioByMouseClick(event, x, y, flags, params):
    global ButtonFlag
    global LookForThisClassName

    if event == cv2.EVENT_LBUTTONDOWN:
        #print(x, y)
        if x1 <= x <= x2 and y1 <= y <= y2:
            print("Click inside the button")

            # record a voice for 3 seconds
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
            sd.wait()  # wait until the recording is finished
            write(audioFileName, fs, myrecording)  # save the audio file

            # extract the text from an audio
            LookForThisClassName = getTextFromAudio()

            if ButtonFlag is False:
                ButtonFlag = True
        else:
            print("Click outside the button")
            ButtonFlag = False

def getTextFromAudio():
    # convert the audio file for Google API
    data, samplerate = soundfile.read(audioFileName)
    soundfile.write('C:/Users/jalaj/OneDrive/Desktop/VoiceBlind/output.wav', data, samplerate, subtype='PCM_16')

    # extract the text
    recognizer = sr.Recognizer()
    jackhammer = sr.AudioFile('C:/Users/jalaj/OneDrive/Desktop/VoiceBlind/output.wav')

    with jackhammer as source:
        audio = recognizer.record(source)

    result = recognizer.recognize_google(audio)

    print(result)
    return result

def speak_detection_message(name, x, y, frame_width, frame_height):
    center_x = x + (frame_width / 2)
    center_y = y + (frame_height / 2)
    width_ratio = center_x / frame_width
    height_ratio = center_y / frame_height

    if width_ratio < 0.4:
        position = "left"
    elif width_ratio > 0.6:
        position = "right"
    else:
        position = "center"

    if height_ratio < 0.4   :
        vertical_position = "top"
    elif height_ratio > 0.6:
        vertical_position = "bottom"
    else:
        vertical_position = "in the"

    message = f"{name} detected {vertical_position} {position} of the camera view"
    engine.say(message)
    engine.runAndWait()

# create our window
cv2.namedWindow("Frame")  # set the same name
cv2.setMouseCallback("Frame", recordAudioByMouseClick)

while True:
    rtn, frame = cap.read()

    # Detect objects
    (class_ids, scores, bboxes) = model.detect(frame)

    # Reset detected count for each frame
    detected_count = 0

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        # draw a rectangle for each detected object
        x, y, width, height = bbox  # x, y is the left upper corner
        name = classesNames[class_id]

        index = LookForThisClassName.find(name)  # look for the text inside a string

        if ButtonFlag is True and index > 0:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (130, 50, 50), 3)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (120, 50, 50), 2)

            # Speak the detection message
            if not detected_flag:
                detected_count += 1
                speak_detection_message(name, x, y, frame.shape[1], frame.shape[0])
                detected_flag = True
                detected_time = time.time()  # Record the time when an object is detected

    # Reset the flag after a certain time (7 seconds in this example)
    if detected_flag:
        if time.time() - detected_time >= 7:
            detected_flag = False

    # draw a "Button"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (153, 0, 0), -1)  # -1 is a filled rectangle
    cv2.putText(frame, "Click for record - 3 seconds", (40, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # white color

    # Display the count of detected objects
    cv2.putText(frame, f"Detected Objects: {detected_count}", (40, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
