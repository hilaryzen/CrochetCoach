import speech_recognition as sr

import PySimpleGUI as sg
import cv2
import numpy as np
import sys
from sys import exit as exit
from gtts import gTTS
from pygame import mixer
import time
import os
from PIL import Image, ImageTk, ImageSequence
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

definitions = {'sc': 'single crochet', 'dc': 'double crochet', 'ch': 'chain', 'sl st': 'slip stitch'}
tutorials = {'sc': 'SingleCrochet.gif', 'single crochet': 'SingleCrochet.gif', 'dc': 'DoubleCrochet.gif', 'double crochet': 'DoubleCrochet.gif', 'ch': 'Chain.gif', 'chain': 'Chain.gif'}

voice_instructions = "\nVoice Commands:\n\"next\": move to the next row in your project\n\"back\": move to the previous row in your project\n\"define <term>\": the coach will read the common definition of a crochet abbreviation\n\tavailable for ch, sc, dc\n\"learn <term>\": the coach will play a GIF tutorial of the requested stitch\n\tavailable for ch, sc, dc\n\"exit\": leave the tutorial and return to webcam\n"

class landmarker_and_result():
   def __init__(self):
      self.result = mp.tasks.vision.HandLandmarkerResult
      self.landmarker = mp.tasks.vision.HandLandmarker
      self.createLandmarker()
   
   def createLandmarker(self):
      # callback function
      def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
         self.result = result

      # HandLandmarkerOptions (details here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream)
      options = mp.tasks.vision.HandLandmarkerOptions( 
         base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"), # path to model
         running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
         num_hands = 2, # track both hands
         min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
         min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
         min_tracking_confidence = 0.3, # lower than value to get predictions more often
         result_callback=update_result)
      
      # initialize landmarker
      self.landmarker = self.landmarker.create_from_options(options)
   
   def detect_async(self, frame):
      # convert np frame to mp image
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      # detect landmarks
      self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

   def close(self):
      # close landmarker
      self.landmarker.close()

def format_pattern(rows):
    formatted = ""
    start = 1
    for r in rows:
        formatted = formatted + f"Row {start}: {r}\n"
        start += 1
    return formatted

def format_row(number, text):
    return f"Row {number}: {text}"

def ColumnFixedSize(layout, size=(None, None), *args, **kwargs):
    # print(size)
    # An addition column is needed to wrap the column with the Sizers because the colors will not be set on the space the sizers take
    return sg.Column([[sg.Column([[sg.Sizer(0,size[1]-1), sg.Column([[sg.Sizer(size[0]-2,0)]] + layout, *args, **kwargs, pad=(0,0))]], *args, **kwargs)]],pad=(0,0))

def read(text):
    mixer.init()
    tts = gTTS(text=text, lang='en',slow=False)
    tts.save('speech0.mp3')
    # playback the speech
    mixer.music.load('speech0.mp3')
    mixer.music.play()
    # wait for playback to end
    while mixer.music.get_busy():
        time.sleep(.1)
    mixer.stop()
    try:
        os.remove('speech0.mp3')
    except:
        pass

def main():

    sg.theme('Black')

    
    row_position = 1
    pattern_rows = []

    # define the window layout
    pattern_layout = [[sg.Text('Row 1:', font='Helvetica 24'), sg.Input(key='r1', font='Helvetica 24')],
                      [sg.Text('# of ch:', font='Helvetica 20'), sg.Input(key='ch1', size=(5, 1), font='Helvetica 20'), sg.Text('# of sc:', font='Helvetica 20'), sg.Input(key='sc1', size=(5, 1), font='Helvetica 20'), sg.Text('# of dc:', font='Helvetica 20'), sg.Input(key='dc1', size=(5, 1), font='Helvetica 20')],
                      [sg.Text('Row 2:', font='Helvetica 24'), sg.Input(key='r2', font='Helvetica 24')],
                      [sg.Text('# of ch:', font='Helvetica 20'), sg.Input(key='ch2', size=(5, 1), font='Helvetica 20'), sg.Text('# of sc:', font='Helvetica 20'), sg.Input(key='sc2', size=(5, 1), font='Helvetica 20'), sg.Text('# of dc:', font='Helvetica 20'), sg.Input(key='dc2', size=(5, 1), font='Helvetica 20')],
                      [sg.Text('Row 3:', font='Helvetica 24'), sg.Input(key='r3', font='Helvetica 24')],
                      [sg.Text('# of ch:', font='Helvetica 20'), sg.Input(key='ch3', size=(5, 1), font='Helvetica 20'), sg.Text('# of sc:', font='Helvetica 20'), sg.Input(key='sc3', size=(5, 1), font='Helvetica 20'), sg.Text('# of dc:', font='Helvetica 20'), sg.Input(key='dc3', size=(5, 1), font='Helvetica 20')],
                      [sg.Button('Start', size=(10, 1), font='Helvetica 24')]]
    left_column_layout = [[sg.Text("Pattern", size=(20, 15), font='Helvetica 24', key='pattern-text')], 
                          [sg.Text(voice_instructions, size=(35, 15), font='Helvetica 20', key='instructions')]]
    crochet_layout = [[sg.Text("Not started", size=(80, 1), justification='center', font='Helvetica 30', key='row-text')],
              [ColumnFixedSize(left_column_layout, size=(30,50)), 
               sg.Image(filename='', key='image', size=(50, 50))]]
    layout = [[sg.Column(pattern_layout, key='pattern-input'), sg.Column(crochet_layout, visible=False, key='crochet-ui')]]

    # layout = [[sg.Text('OpenCV Demo', size=(80, 1), justification='center', font='Helvetica 20')],
    #           [sg.Text(format_pattern(pattern_rows), size=(30, 40)), sg.Image(filename='', key='image')]]

    # create the window and show it without the plot
    window = sg.Window('Crochet Coach',
                       layout,
                       size=(1400, 1000))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv2.VideoCapture(0)
    hand_landmarker = landmarker_and_result()
    twists_per_row = []
    recording = False
    on_pattern_entry = True
    running_tutorial = False
    running_gif_name = ''
    total_twists = 0
    twisting = False

    def go_to_next_row():
        nonlocal row_position
        nonlocal recording
        nonlocal total_twists
        total_twists = 0
        if row_position == len(pattern_rows):
            nonlocal on_pattern_entry
            window['row-text'].update("Finished")
            read("Finished")
            
            window['pattern-input'].update(visible=True)
            window['crochet-ui'].update(visible=False)
            on_pattern_entry = True
            recording = False
        else:
            row_position += 1
            window['row-text'].update(format_row(row_position, pattern_rows[row_position - 1]))
            read(format_row(row_position, pattern_rows[row_position - 1]))

    def go_to_previous_row():
        nonlocal row_position
        if row_position > 1:
            row_position -= 1
        total_twists = 0
        window['row-text'].update(format_row(row_position, pattern_rows[row_position - 1]))
        read(format_row(row_position, pattern_rows[row_position - 1]))



    # this is called from the background thread
    def callback(recognizer, audio):
        # received audio data, now we'll recognize it using Google Speech Recognition
        nonlocal row_position
        WIT_AI_KEY = "QORRVL4NM4OV4GNGTTXZUKEVVZOPUD4C"
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            value = recognizer.recognize_google(audio)
            print("Google Speech Recognition thinks you said " + value)
            user_command = value.lower().split(' ', 1)
            if user_command[0] == 'define':
                if user_command[1] in definitions.keys():
                    read(definitions[user_command[1]])
                else:
                    read("Could not find definition for " + user_command[1])
            elif user_command[0] == 'learn':
                if user_command[1] in tutorials.keys():
                    nonlocal recording
                    nonlocal running_tutorial
                    nonlocal running_gif_name
                    recording = False
                    running_tutorial = True
                    running_gif_name = tutorials[user_command[1]]
                else:
                    read("Could not find tutorial for " + user_command[1])
            elif user_command[0] == 'exit':
                running_tutorial = False
                recording = True
            elif user_command[0] == 'next':
                # go to next row
                go_to_next_row()
            elif user_command[0] == 'back':
                go_to_previous_row()
            # print("Google Speech Recognition thinks you said " + recognizer.recognize_google(audio))
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    r = sr.Recognizer()
    m = sr.Microphone()
    with m as source:
        r.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening

    # start listening in the background (note that we don't have to do this inside a `with` statement)
    stop_listening = r.listen_in_background(m, callback)

    def detect_twist(detection_result: mp.tasks.vision.HandLandmarkerResult):
        # print(detection_result)
        nonlocal total_twists
        nonlocal twisting
        try:
            if detection_result.hand_landmarks == []:
                # print("none")
                return False
            else:
                hand_landmarks_list = detection_result.hand_landmarks
                pinky_tip = hand_landmarks_list[0][20]
                index_tip = hand_landmarks_list[0][8]
                if pinky_tip.y > index_tip.y:
                    if not twisting:
                        total_twists += 1
                        twisting = True
                        print(total_twists)
                        return True
                else:
                    twisting = False
                    return False
        except Exception as error:
            # print(error)
            return False

    while True:
        event, values = window.read(timeout=100)
        if event == sg.WIN_CLOSED:
            stop_listening(wait_for_stop=False)
            return
        
        if on_pattern_entry:
            if event == 'Start':
                on_pattern_entry = False
                pattern_rows = [values['r1'], values['r2'], values['r3']]
                twists_per_row = [int(values['ch1']) + 3 * int(values['sc1']) + 6 * int(values['dc1']), int(values['ch2']) + 3 * int(values['sc2']) + 6 * int(values['dc2']), int(values['ch3']) + 3 * int(values['sc3']) + 6 * int(values['dc3'])]
                print(twists_per_row)

                window['pattern-input'].update(visible=False)
                window['crochet-ui'].update(visible=True)
                window['pattern-text'].update(format_pattern(pattern_rows))
                window['row-text'].update(format_row(row_position, pattern_rows[row_position - 1]))
                recording = True
                event, values = window.read(timeout=100)
                read(format_row(row_position, pattern_rows[row_position - 1]))
        
        # if not on_pattern_entry:
        #     if event == 'Exit' or event == sg.WIN_CLOSED:
        #         stop_listening(wait_for_stop=False)
        #         return

        if running_tutorial:
            # sg.PopupAnimated(tutorials[user_command[1]], no_titlebar=False, time_between_frames=100)
            for frame in ImageSequence.Iterator(Image.open(running_gif_name)):
                event, values = window.read(timeout=100)
                if event == sg.WIN_CLOSED:
                    stop_listening(wait_for_stop=False)
                    exit(0)
                if running_tutorial:
                    window['image'].update(data=ImageTk.PhotoImage(frame) )
                else:
                    break

        if recording:
            ret, frame = cap.read()
            
            hand_landmarker.detect_async(frame)
            detect_twist(hand_landmarker.result)
            if total_twists == twists_per_row[row_position - 1]:
                read("Coach detected a finished row")
                go_to_next_row()
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            # print(sys.getsizeof(imgbytes))
            window['image'].update(data=imgbytes)


main()