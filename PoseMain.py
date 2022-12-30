# this program was written in Pycharm and uses imports like OpenCV, Mediapipe and FastDTW as well as spicy which is is required
# in order to run FastDTW and give the required outputs7
import cv2
import PoseEstimationModule as pm
import PoseEstimationModule as pm2
from scipy.spatial.distance import cosine
from fastdtw import fastdtw

user = cv2.VideoCapture('Video/Video_Final1.mp4')  # the video input for the user
pro = cv2.VideoCapture('Video/Video_Final2.mp4')  # the video input for the pro
userDetect = pm.poseDector()
proDetect = pm2.poseDector()

correctPose = 0
incorrectPose = 0
while True:
    userSuccess, userImg = user.read()
    proSuccess, proImg = pro.read()
    if userSuccess:
        userImg = userDetect.findPose(userImg)
        userList = userDetect.findPosition(userImg)
        proImg = proDetect.findPose(proImg)
        proList = proDetect.findPosition(proImg)
        # print(userList)
        # print(proList)
        if len(userList) == 0 or len(proList) == 0:
            print("issue with recording at this frame")
        else:
            error, _ = fastdtw(userList, proList, dist=cosine)
            # print(error)
            if error < 0.09:
                correctPose = correctPose + 1
            else:
                incorrectPose = incorrectPose + 1

        cv2.imshow("User", userImg)
        cv2.imshow("Pro", proImg)
        cv2.waitKey(1)  # 1 millisecond delay
    else:
        break

print("Code ran!")
if correctPose > incorrectPose:
    print("Congrats most of your movement were correct")
else:
    print("Sorry but you made some errors, redo and practise!")
print("the number of incorrect movements made were: " + str(incorrectPose) + " Out of " + str(incorrectPose + correctPose))
