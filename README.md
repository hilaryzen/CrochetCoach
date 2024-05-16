# CrochetCoach
6.8510 Final Project
Hilary Zen

All code for the coach interface is in main.py
The MediaPipe hand landmark detection model is in hand_landmarker.task
There are three tutorial GIFs used by the system: Chain.gif, DoubleCrochet.gif, SingleCrochet.gif
requirements.txt contains all the libraries needed to run

The Crochet Coach runs on Python 3.12.3
To clone the repo and setup, run:
```
git clone https://github.com/hilaryzen/CrochetCoach.git
cd CrochetCoach
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run the coach, use:
```
python main.py
```