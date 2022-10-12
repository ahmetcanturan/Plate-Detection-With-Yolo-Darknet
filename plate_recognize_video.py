# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 07:06:11 2021

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 01:12:33 2021

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 19:06:01 2021

@author: HP
"""
## nesne tanıyarak aynı nesnenin birden fazla kare içine alınmasını engelleyeceğiz
# confidenceleri listeliyeceğiz ve güvenirlik oranları en yüksek bounding boxları çizdireceğiz


import cv2
import numpy as np

cap=cv2.VideoCapture("plate_video.mp4")

labels = ["Plaka"] 
colors=["0,255,0","0,0,255","255,0,0","100,0,20","255,255,0"]
colors=[np.array(color.split(",")).astype("int") for color in colors]
    # string değerleri int değerlere çevirdik simdide tek bir matriste toplayalım
colors=np.array(colors)
    # 5 ten fazla eleman olduğunda matris büyütme yapalım.
colors=np.tile(colors,(18,1)) #18 kez büyüttük  
model=cv2.dnn.readNetFromDarknet("C:/Users/HP/Yolo Projects/plate/plate_yolov4.cfg","C:/Users/HP/Yolo Projects/plate/plate_yolov4_last.weights")
layers=model.getLayerNames()
output_layer=[layers[layer[0]-1]for layer in model.getUnconnectedOutLayers()]

while True:
    ret,frame=cap.read()
    frame_width=frame.shape[1]
    frame_height=frame.shape[0]
    frame_blob=cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB=True,crop=False)
    ids_list=[]
    boxes_list=[]
    confidence_list=[] 
    model.setInput(frame_blob) 
    detection_layers=model.forward(output_layer) # tespit edilen katmanlar forward yollamak
    for detection_layer in detection_layers: # fotografta 3 farklı nesne cinsi tespit etti yani len(detection_layers)=3
        for object_detection in detection_layer:
            #ilk 5 değer bounding box ile ilgili
            #öncelikle 5 den sonraki değerler ile yani 
            #güven skorlarıyla ilgilenicez
            scores=object_detection[5:]
            # en büyük değer tahmin edilen id olacak
            predicted_id=np.argmax(scores) # en büyük değerin indisi
            confidence=scores[predicted_id] # güven scoruna bu değişkenden ulaşıcaz
            if confidence > 0.20: # güven skoru %30dan büyük ise bounding box çizdir
                label=labels[predicted_id]# en büyük tahmin indisi labelin o indisine karşılık gelir
                bounding_box=object_detection[0:4]*np.array([frame_width,frame_height,frame_width,frame_height])
                (box_center_x,box_center_y,box_width,box_height)=bounding_box.astype("int")
                # dikdörtgenin başlangış noktalarını belirleyeceğiz
                start_x=int(box_center_x-(box_width/2))
                start_y=int(box_center_y-(box_height/2))
                
                #########Non-Maximum-Supression - Operation-2 ###########
                ids_list.append(predicted_id)
                confidence_list.append(float(confidence))
                boxes_list.append((start_x,start_y,int(box_width),int(box_height)))
    
                ##########End of operation 2################################
            
 #########Non-Maximum-Supression - Operation-3 ###########
# en yüksek güven skorlu boxesları döndürür.
    max_ids=cv2.dnn.NMSBoxes(boxes_list,confidence_list,0.5,0.4)
    for max_id in max_ids:
         max_class_id=max_id[0]
         box=boxes_list[max_class_id]
         start_x=box[0]
         start_y=box[1]
         box_width=box[2]
         box_height=box[3]
         predicted_id=ids_list[max_class_id]
         label=labels[predicted_id]
         confidence=confidence_list[max_class_id]
    
    
    ##########End of operation 3################################
         end_x=start_x + box_width
         end_y=start_y + box_height
    
         box_color=colors[predicted_id]
         box_color=[int(each) for each in box_color]
                
         label="{}: {:.2f}%".format(label,confidence*100)
         print("Predection Object {}".format(label))            
         cv2.rectangle(frame,(start_x,start_y),(end_x,end_y),box_color,2)
         cv2.putText(frame,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,2)
                           
    cv2.imshow("Detections",frame)
    if cv2.waitKey(1)& 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()            
         