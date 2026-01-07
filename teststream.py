#!/usr/bin/env python3
import cv2
import time
cap = cv2.VideoCapture("http://holoscope2.sfs.uwm.edu:81/stream")

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
framesleep = 10
frameno = 0
start_time=0
while True:
    last_time = start_time
    start_time = time.perf_counter()
    if frameno > 0:
        elapsed_time = start_time - last_time
        eff_fps = 1.0 / elapsed_time
    ret, frame = cap.read()
    end_cap_time = time.perf_counter()
    elapsed_cap_time = end_cap_time - start_time
    if not ret:
        print("Error: Could not read frame.")
        break
    if elapsed_cap_time > 0.5:
        framesleep = max(1, framesleep + 5)
        frameno = 0
        last_time = 0
        first_time = 0

    frameno += 1
    if frameno > 1:
        cv2.putText(frame, f"{frameno}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"{frameno}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"{elapsed_cap_time:0.3f}", (12, 132), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"{elapsed_cap_time:0.3f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255, 0), 2)
        cv2.putText(frame, f"{framesleep}", (12, 232), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 0), 2)
        cv2.putText(frame, f"{framesleep}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"{eff_fps}", (112, 232), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 0), 2)
        cv2.putText(frame, f"{eff_fps}", (110, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('Video Stream', frame)

    k=cv2.waitKey(framesleep)
    if k & 0xFF == ord('q'):
        break
    if k & 0xFF == 82:  # Up arrow key
        framesleep += 5
    if k & 0xFF == 84:  # Down arrow key
        framesleep = max(1, framesleep - 5)

cap.release()

