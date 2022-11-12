# TODO: argparse
# TODO: main function
from multiprocessing import Process, Queue
from datetime import datetime

import cv2


def image_vis(taskqueue, width, height, fps, frames_per_file):

    # 指定影片編碼
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'H264')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = None

    while True:
        # 從工作佇列取得影像
        image, frame_counter = taskqueue.get()

        # 若沒有影像則終止迴圈
        if image is None: break

        if frame_counter % frames_per_file == 0:

            if writer: writer.release()

            # 建立 VideoWriter 物件（以數字編號）
            # index = int(frame_counter // frames_per_file)
            # writer = cv2.VideoWriter(f'output-{index}.mp4', fourcc, fps, (width, height))

            # 建立 VideoWriter 物件（以時間命名）
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
            writer = cv2.VideoWriter(f'videos/output-{timestamp}.mp4', fourcc, fps, (width, height))

        # 儲存影像
        cv2.putText(image, 'Predict', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
        for i in range(80):
            if i %2 == 0:
                color = (0, 255, 0)
            else:
                color = (255, 255, 255)
            cv2.rectangle(image, (60+i, 200), (61+i, 215),
                            color, -1)
        writer.write(image)

    # 釋放資源
    writer.release()


if __name__ == '__main__':

    # 開啟 RTSP 串流
    LAB_RTSP = 'rtsp://root:a1s2d3f4@192.168.50.161:554/live.sdp'
    f = r'C:\Users\test\Desktop\Leon\Datasets\Breakfast\BreakfastII_15fps_qvga_sync\P03\cam01\P03_cereals.avi'
    
    vidCap = cv2.VideoCapture(f)

    # 取得影像的尺寸大小
    width = int(vidCap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 取得影格率
    fps = vidCap.get(cv2.CAP_PROP_FPS)

    # 建立工作佇列
    taskqueue = Queue()

    # 計數器
    frame_counter = 0

    total_time = 100 * 3600 # 30 seconds per process
    file_time = 10 * 60 # 10 seconds per file

    # 總錄製幀數（30 秒鐘）
    total_frames = fps * total_time

    # 每個檔案的幀數（10 秒鐘）
    frames_per_file = fps * file_time

    # 建立並執行工作行程
    proc = Process(target=image_vis, args=(taskqueue, width, height, fps, frames_per_file))
    proc.start()

    while frame_counter < total_frames:
        # 從 RTSP 串流讀取一張影像
        ret, image = vidCap.read()

        if ret:
            # 將影像放入工作佇列
            taskqueue.put((image, frame_counter))
            frame_counter += 1
        else:
            # 若沒有影像跳出迴圈
            break

    # 傳入 None 終止工作行程
    taskqueue.put((None, None))

    # 等待工作行程結束
    proc.join()

    # 釋放資源
    vidCap.release()