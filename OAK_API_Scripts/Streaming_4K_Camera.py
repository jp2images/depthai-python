from pathlib import Path
import numpy as np
import cv2  # opencv - display the video stream
import depthai  # access the camera and its data packets

device = depthai.Device('', False)
# Create the pipeline using the 'color' stream, establishing the first connection to the device.
pipeline = device.create_pipeline(config={'streams': ['color'],    
                                          'ai': {"blob_file": str(Path('./mobilenet-ssd/mobilenet-ssd.blob').resolve().absolute()),
                                                 "blob_file_config": str(Path('./mobilenet-ssd/mobilenet-ssd.json').resolve().absolute()),}
                                         })

scale = 1.0

if pipeline is None:   
    raise RuntimeError('Pipeline creation failed!')
    
while True:    # Retrieve data packets from the device.   # A data packet contains the video frame data.    
    nnet_packets, data_packets = pipeline.get_available_nnet_and_data_packets()


    for packet in data_packets:   
        if packet.stream_name == 'color':  
            window_name = packet.stream_name
            packetData = packet.getData()    
            meta = packet.getMetadata()
            w = meta.getFrameWidth()
            h = meta.getFrameHeight()
            yuv420p = packetData.reshape((h * 3 // 2, w))
            bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            cv2.putText(bgr, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            #cv2.putText(bgr, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.imshow(window_name, bgr)        
                    
    if cv2.waitKey(1) == ord('q'):
        break

# The pipeline object should be deleted after exiting the loop. Otherwise device will continue working.
# This is required if you are going to add code after exiting the loop.
del pipeline
del device            



