#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
import csv


def down_scale(image):
    height, width, _ = image.shape
    new_dim = (int(width / 2), int(height / 2))  # (width, height)
    new_img = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    return new_img


def add_mask(points,image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # roi_points = np.array([[0, 360], [0, 298], [298, 205], [373, 217], [640, 329], [640, 360]], dtype=np.int32)
    roi_points = np.array(points, dtype = np.int32)
    cv2.fillConvexPoly(mask, roi_points, 255)
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    return masked_img


def dense_optical_flow(prev_img, image):
    bgr = None
    hsv = np.zeros_like(image)  # hsv.shape:(360, 640, 3)
    hsv[..., 1] = 255  # color scale
    if prev_img is None:
        print('Initializing a prev_image...')
        prev_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # prev_img.shape:(360, 640)
    else:
        next_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, None,
                                            pyr_scale=0.5,
                                            levels=3,
                                            winsize=15,
                                            iterations=3,
                                            poly_n=5,
                                            poly_sigma=1.1,
                                            flags=0)  # flow.shape:(360, 640, 2)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue, corresponds to direction
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value -> Magnitude
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        prev_img = next_img
    return bgr, prev_img


def extra_point(aol,img):
    points = SetPoints(str(aol), img)
    temp = np.array(points)
    pts = temp.reshape(4, 2).astype(np.float32)
    return pts

def read_img(path):
    image = cv2.imread(path)
    w=image.shape[0]
    h=image.shape[1]
    ratio = image.shape[0] / 500.0  # 比例
    orig = image.copy()
    image = imutils.resize(image, height=500)
    return image

def extra_point(aol,img):
    points = SetPoints(str(aol), img)
    temp = np.array(points)
    pts = temp.reshape(4, 2).astype(np.float32)
    return pts

def read_csv(path, name):
    with open(path,"r",newline="", encoding="utf-8-sig", errors="ignore") as f:
        csv_write = csv.reader(f)
        for i,row in enumerate(csv_write):
            if str(name) in row:
                return row[2], row[3]
            else:
                continue

def main():
    
    # 调用摄像头
    # capture=cv2.VideoCapture(0) # capture=cv2.VideoCapture("1.mp4")
    #----------------------------cut video according to the csv file
    path = "./a.mp4"
    # file_name = path.split("/")[-1].split(".")[-2]
    # # print(file_name)
    # csv_path = "./sth_4_1/ground_truth.csv"
    # time1, time2 = read_csv(csv_path, file_name)
    # old_capture = cv2.VideoCapture(path)
    # fps_old = 30
    # size_old = (1280, 720)
    # new_path = "./sth_4_1/"+file_name+".avi"
    # videoWriter = cv2.VideoWriter(new_path,cv2.VideoWriter_fourcc('X','V','I','D'),fps_old,size_old)

    # ii = 0
    # while True:
    #     success,frame = old_capture.read()
    #     if success:
    #         ii += 1
    #         print("ii = ",ii)
    #         if ii>=int(time1) and ii<int(time2):
    #             videoWriter.write(frame)
    #     else:
    #         print("transform finished")
    #         break
    #----------------------------
    
    capture = cv2.VideoCapture(path)
    fps = 0.0

    yolo = YOLO()
    while(True):
        t1 = time.time()
        # 读取某一帧
        ref,frame=capture.read()

        if ref == True:
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            #############------------------------
            # mask_point = np.array([[0, 426], [173, 384], [376, 354], [400, 363], [633, 327], [717, 358], [789, 393], [824, 416], [867, 458], [911, 500], [989, 537], [1081, 583], [1150, 623], [1222, 659], [1279, 701], [1278, 718], [1, 718]], dtype=np.int32)
            # frame = add_mask(mask_point, frame)
            # #实际坐标点和提取的角点必须一一对应呀，
            # point1 = np.array([[90, 555], [661, 399], [1104, 641], [65, 674]], dtype=np.int32)
            # point1 = point1.reshape(4,2).astype(np.float32)
            # point2 = np.array([[188, 132], [510, 170], [472, 498], [154, 221]], dtype=np.int32)
            # point2 = point2.reshape(4,2).astype(np.float32)
            # M = cv2.getPerspectiveTransform(point1,point2)
            # frame = cv2.warpPerspective(frame,M,(frame.shape[0],frame.shape[1]))

            #############------------------------



            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))

            # 进行检测
            print(yolo.new_detect(frame))
            frame = np.array(yolo.detect_image(frame))
            # #实际坐标点和提取的角点必须一一对应呀，
            # point1 = np.array([[90, 555], [661, 399], [1104, 641], [65, 674]], dtype=np.int32)
            # point1 = point1.reshape(4,2).astype(np.float32)
            # point2 = np.array([[188, 132], [510, 170], [472, 498], [154, 221]], dtype=np.int32)
            # point2 = point2.reshape(4,2).astype(np.float32)
            # M = cv2.getPerspectiveTransform(point1,point2)
            # frame = cv2.warpPerspective(frame,M,(frame.shape[0],frame.shape[1]))

            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            #----- write fps on the frame
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video",frame)



        else:
            break
        c= cv2.waitKey(30) & 0xff 
        if c==27:
            capture.release()
            break

if __name__ == "__main__":
    main()