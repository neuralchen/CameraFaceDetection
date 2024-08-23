import cv2
from insightface_func.face_detect_crop_single import Face_detect

if __name__ == "__main__":
    cap = cv2.VideoCapture(1) 
    if cap.isOpened():
        print("found camera 1")
    else:
        print("No Camera Found!")
        exit()
    # 以下set若相机不支持该分辨率，即使执行也无效
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    mode   = "ffhq" # 人脸检测后的crop方式

    detection = Face_detect(name='antelope', root='./insightface_func/models')
    detection.prepare(ctx_id = 0, det_thresh=0.6,\
                        det_size=(640,640), mode = mode)
    
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read() 
        detect_results = detection.get(frame, crop_size=512)
        
        if detect_results is not None:

            face = detect_results[0][0]
            cv2.resizeWindow("window", 1920, 1080)
            cv2.imshow("window", face) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        
    cap.release()  
    cv2.destroyAllWindows() 
