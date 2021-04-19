import pywintypes
import win32gui as w
while True:
    running_app = w.GetWindowText(w.GetForegroundWindow())
    print(running_app)
