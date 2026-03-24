from ultralytics import YOLO
import cv2


model = YOLO("./yolo26s-pose.pt")

results = model("https://www.shutterstock.com/image-photo/portrait-asian-business-person-crossed-260nw-2603900035.jpg")  # predict on an image

# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)

    frame = result.plot()
    cv2.imshow("YOLO Pose Visualization", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
