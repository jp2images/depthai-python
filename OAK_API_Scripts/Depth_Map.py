from pathlib import Path
import numpy as np
import cv2  # opencv - display the video stream
import depthai  # access the camera and its data packets

device = depthai.Device('', False)
# Create the pipeline using the 'depth' stream, establishing the first connection to the device.
pipeline = device.create_pipeline(config={'streams': ['depth'],    
                                          'ai': {"blob_file": str(Path('./mobilenet-ssd/mobilenet-ssd.blob').resolve().absolute()),
                                                 "blob_file_config": str(Path('./mobilenet-ssd/mobilenet-ssd.json').resolve().absolute()),}
                                         })
if pipeline is None:   
    raise RuntimeError('Pipeline creation failed!')

frame_count = {}
frame_count_prev = {}

stream_windows = ['depth']

for w in stream_windows:
    frame_count[w] = 0
    frame_count_prev[w] = 0
    
detections = []

disparity_confidence_threshold = 200

def on_trackbar_change(value):
        device.send_disparity_confidence_threshold(value)
        return

for stream in stream_windows:
    if stream in ["disparity", "disparity_color", "depth"]:
        cv2.namedWindow(stream)
        trackbar_name = 'Disparity confidence'
        conf_thr_slider_min = 0
        conf_thr_slider_max = 255
        cv2.createTrackbar(trackbar_name, stream, conf_thr_slider_min, conf_thr_slider_max, on_trackbar_change)
        cv2.setTrackbarPos(trackbar_name, stream, disparity_confidence_threshold)
    
t_start = time()  

while True:    # Retrieve data packets from the device.   # A data packet contains the video frame data.    
    nnet_packets, data_packets = pipeline.get_available_nnet_and_data_packets()

    for nnet_packet in nnet_packets:
        detections = list(nnet_packet.getDetectedObjects())
        print(detections)

    for packet in data_packets:   # By default, DepthAI adds other streams (notably 'meta_2dh'). Only process `depth`.                
        window_name = packet.stream_name
        if packet.stream_name == 'depth':    
          
            frame = packet.getData()                    
            frame = (65535 // frame).astype(np.uint8)
            # colorize depth map
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))                
            cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)           
            cv2.imshow(packet.stream_name, frame)   

        frame_count[window_name] += 1    
        
        
    t_curr = time()
    if t_start + 1.0 < t_curr:
        t_start = t_curr
        for w in stream_windows:
            frame_count_prev[w] = frame_count[w]
            frame_count[w] = 0            
                    
    if cv2.waitKey(1) == ord('q'):
        break

# The pipeline object should be deleted after exiting the loop. Otherwise device will continue working.
# This is required if you are going to add code after exiting the loop.
del pipeline
del device