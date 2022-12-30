import cv2
import mediapipe as mp


class poseDector():

    def __init__(self, mode=False, complexity=1, smoothLandmark=True, enableSegmentation=False, smoothSegmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smoothLandmark = smoothLandmark
        self.enableSegmentation = enableSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils  # used to draw the skeleton frame work on the video
        self.mpPose = mp.solutions.pose  # extracts the pose estimation from media pipe
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smoothSegmentation, self.smoothLandmark,
                                     self.enableSegmentation,
                                     self.detectionCon, self.trackCon)

    # method to draw the skeleton
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)  # drawing the skeleton structure
        return img

    # method to find the points on the skeleton
    def findPosition(self, img):
        poselist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #if id == 11 or id == 12 or id == 13 or id == 14 or id == 23 or id == 24 or id == 25 or id == 26:
                    #poselist.append([cx, cy])
                poselist.append([cx, cy])
        return poselist
