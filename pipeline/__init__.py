"""Detection pipeline package (Process 1 - offline, heavy CV deps).

geometry.py and sessions.py are pure-Python and importable without torch/cv2;
detect.py and reid.py lazy-import their heavy dependencies.
"""
