# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 19:06:01 2021

@author: HP
"""
## nesne tanıyarak aynı nesnenin birden fazla kare içine alınmasını engelleyeceğiz
# confidenceleri listeliyeceğiz ve güvenirlik oranları en yüksek bounding boxları çizdireceğiz

#%% 1. Bölüm

import cv2
import numpy as np

img=cv2.imread("img.png")
img2=img.copy()
img_width=img.shape[1]
img_height=img.shape[0]
#%% 2. Bölüm

# yolo algoritmasının fotoğrafı okuması için 4 boyutlu denilen blob formata çevrilmesi gerekiyor.
img_blob=cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)
# 1/255-Resmin yeniden boyutlandırılması için gerekli olan scale faktörü
# 416lık bir model indirdiğimiz için yeni resmi 416 ya 416 piksel yaptık
# swapRB=True Resmi bgr formattan rgb formata çevirmemiz gerekiyor 
# crop= False resmin kırpılmasını istemedik-Kırpılmasını istediğimizde sırt çantasını buluyor

labels = ["Plaka","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
          "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
          "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
          "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
          "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
          "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
          "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
          "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
          "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
          "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

colors=["0,255,0","0,0,255","255,0,0","100,0,20","255,255,0"]
colors=[np.array(color.split(",")).astype("int") for color in colors]
# string değerleri int değerlere çevirdik simdide tek bir matriste toplayalım
colors=np.array(colors)
# 5 ten fazla eleman olduğunda matris büyütme yapalım.
colors=np.tile(colors,(18,1)) #18 kez büyüttük  

#%% 3. Bölüm
#config ve weight dosyalarımızı models değişkenine ekledik
model=cv2.dnn.readNetFromDarknet("C:/Users/HP/Yolo Projects/plate/plate_yolov4.cfg","C:/Users/HP/Yolo Projects/plate/plate_yolov4_last.weights")
#model değişleni içerisinden bazı layersleri çekelim Katmanları çekelim
layers=model.getLayerNames()
# layers değiskenindeki bir sürü katmandan çıktı katmanını çekmemiz gerekiyor.
#model.getUn.. komutu çıkış layerlerini döndürür indisler 0 dan başlaığı için 
# döndürdüğü sayılardan 1 çıkarıyoruz
output_layer=[layers[layer[0]-1]for layer in model.getUnconnectedOutLayers()] 

# 4 boyutlu resmimizi artık modele okutmamız gerekiyor 
model.setInput(img_blob)

# çıktı katmanlarımızı modele yolluyalım
detection_layers=model.forward(output_layer) # tespit edilen katmanlar forward yollamak

#########Non-Maximum-Supression - Operation-1 ###########
ids_list=[]
boxes_list=[]
confidence_list=[]
##########End of operation 1################################

#%% 4. Bölüm

for detection_layer in detection_layers: # fotografta 3 farklı nesne cinsi tespit etti yani len(detection_layers)=3
    for object_detection in detection_layer:
        #ilk 5 değer bounding box ile ilgili
        #öncelikle 5 den sonraki değerler ile yani 
        #güven skorlarıyla ilgilenicez
        scores=object_detection[5:]
        # en büyük değer tahmin edilen id olacak
        predicted_id=np.argmax(scores) # en büyük değerin indisi
        confidence=scores[predicted_id] # güven scoruna bu değişkenden ulaşıcaz
        if confidence > 0.01: # güven skoru %30dan büyük ise bounding box çizdir
            label=labels[predicted_id]# en büyük tahmin indisi labelin o indisine karşılık gelir
            bounding_box=object_detection[0:4]*np.array([img_width,img_height,img_width,img_height])
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
plate_kordinat_x=[]
plate_kordinat_y=[]
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
     plate_kordinat_x.append(start_x)
     plate_kordinat_x.append(end_x+5)
     plate_kordinat_y.append(start_y)
     plate_kordinat_y.append(end_y)
     cv2.rectangle(img,(start_x,start_y),(end_x,end_y),box_color,2)
     cv2.putText(img,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,2)

if len(max_ids) != 0:
    for m in range(len(max_ids)):
        im_plate=img2[plate_kordinat_y[m]:plate_kordinat_y[m+1],plate_kordinat_x[m]:plate_kordinat_x[m+1]]
        pencere=str(m+1)+ ". Plaka"
        cv2.imshow(pencere,im_plate)
        cv2.imwrite("C:/Users/HP/Desktop/plaka.jpg",im_plate)
    
          
      
cv2.imshow("Detections",img)
cv2.waitKey()
cv2.destroyAllWindows()            
          