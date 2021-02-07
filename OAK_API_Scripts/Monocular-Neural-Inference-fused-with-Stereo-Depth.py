from pathlib import Path
import numpy as np
import cv2  # opencv - display the video stream
import depthai  # access the camera and its data packets

device = depthai.Device('', False)
# Create the pipeline using the 'previewout, metaout & depth' stream, establishing the first connection to the device.
pipeline = device.create_pipeline(config={'streams': ['previewout','metaout','depth'],    
                                          'ai': {"blob_file": str(Path('./mobilenet-ssd/mobilenet-ssd.blob').resolve().absolute()),
                                                 "blob_file_config": str(Path('./mobilenet-ssd/mobilenet-ssd.json').resolve().absolute()),}
                                         })

labels = np.loadtxt("labels.txt", dtype=str)

if pipeline is None:   
    raise RuntimeError('Pipeline creation failed!')
    
nn2depth = True

nn_depth = device.get_nn_to_depth_bbox_mapping()

def nn_to_depth_coord(x, y, nn2depth):
    x_depth = int(nn2depth['off_x'] + x * nn2depth['max_w'])
    y_depth = int(nn2depth['off_y'] + y * nn2depth['max_h'])
    return x_depth, y_depth
    
    
detections = []
    
while True:    # Retrieve data packets from the device.   # A data packet contains the video frame data.    
    nnet_packets, data_packets = pipeline.get_available_nnet_and_data_packets()

    for nnet_packet in nnet_packets:
        detections = list(nnet_packet.getDetectedObjects())

    for packet in data_packets:   # By default, DepthAI adds other streams (notably 'meta_2dh'). Only process `previewout`.       

        if packet.stream_name == 'previewout':  
            meta = packet.getMetadata()
            camera = meta.getCameraName()
            window_name = 'previewout-' + camera
            data = packet.getData()
            # change shape (3, 300, 300) -> (300, 300, 3)            
            data0 = data[0, :, :]            
            data1 = data[1, :, :]            
            data2 = data[2, :, :]           
            frame = cv2.merge([data0, data1, data2])            

            img_h = frame.shape[0]            
            img_w = frame.shape[1]            

            for detection in detections:                 
                pt1 = int(detection.x_min * img_w), int(detection.y_min * img_h)                 
                pt2 = int(detection.x_max * img_w), int(detection.y_max * img_h)                              
                label = labels[int(detection.label)]                 
                score = int(detection.confidence * 100)   
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)    
                cv2.putText(frame, str(score) + ' ' + label,(pt1[0] + 2, pt1[1] + 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)           

            cv2.imshow(window_name, frame)      

        elif packet.stream_name == 'depth':  # Only process `depth`.
            window_name = packet.stream_name
            frame = packet.getData()  
            cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
            frame = (65535 // frame).astype(np.uint8)
            # colorize depth map
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
             #cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255) 
            if detections is not None:
                for detection in detections:  
                    pt1 = nn_to_depth_coord(detection.x_min, detection.y_min, nn_depth)
                    pt2 = nn_to_depth_coord(detection.x_max, detection.y_max, nn_depth)
                    color = (255, 255, 255) # bgr
                    cv2.rectangle(frame, pt1, pt2, color)
            cv2.imshow(window_name, frame)            
          
                    
    if cv2.waitKey(1) == ord('q'):
        break

# The pipeline object should be deleted after exiting the loop. Otherwise device will continue working.
# This is required if you are going to add code after exiting the loop.
del pipeline 
del device            



