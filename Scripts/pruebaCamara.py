##prueba para detectar camaras con V4L2

import cv2, glob, os
devs = sorted(glob.glob('/dev/video*'))
print("V4L2:", devs)
for p in devs:
    idx = int(os.path.basename(p).replace('video',''))
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    ok = cap.isOpened()
    print(p, "abrió?", ok)
    if ok:
        ret, frame = cap.read()
        print("Frame?", ret, "shape:", None if not ret else frame.shape)
        cap.release()
