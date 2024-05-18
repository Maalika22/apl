<html>
<title>
    Program
</title>
<body>
    <p><pre>
        <h3>pr1</h3>
        from PIL import Image

img=Image.open("butterfly.jpg")
img.show()

width, height=img.size

print(width, height)

print(img.format)

print(img.info)

img1=img.save("xyz.png")

img2=img.rotate(90)

img2.show()

size=(250,250,750,750)

img3=img.crop(size)

img3.show()

RE_SIZE(300,300)

img4=img.resize(RE_SIZE)

img4.show()
<h3>pr2</h3>
from PIL import Image
from matplotlib import pyplot as plt
img1=Image.open("butterfly.jpg")
img2=Image.open("butterfly_2.jpeg")
img1.paste(img2,(200,250))
img1.show()
histogram=img1.histogram()
plt.hist(histogram,bins=len(histogram))
plt.show()
img3=img1.transpose(Image.FLIP_LEFT_RIGHT)
img3.show()
red,green,blue=img1.split()
zero=red.point(lambda _:0)
red=Image.merge("RGB",(red,zero,zero))
green=Image.merge("RGB",(zero,green,zero))
blue=Image.merge("RGB",(zero,zero,blue))
red.show()
green.show()
blue.show()
img1.save("abc.bmp")
img2.thumbnail((100,100))
img2.save("thumbnal.png")
                
<h3>erosion</h3>
import cv2
import numpy as np
img = cv2.imread("butterfly.jpg",1)
#create kernel matrix of ones .. for morphological transformations
kernel = np.ones((5,5) , dtype="uint8")
#erode(remove white noises) in binary image
img_erosion = cv2.erode(img, kernel, iterations=1)
#dilate(increases white region) in binar image
img_dilation = cv2.dilate (img, kernel, iterations=1)
cv2.imshow("input", img)
cv2.imshow("erosion",img_erosion)
cv2.imshow("dilation", img_dilation)
cv2.waitKey(0)
<h3>colour</h3>
import cv2
import numpy as np

def nothing(x):
    pass

# Creating a window with black image
img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')

# Creating trackbars for red, green, blue color change
# nothing is call back function
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)

while True:
    # Show image
    cv2.imshow('image', img)
    
    # Get current positions of all three trackbars
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    
    # Display color mixture
    #img[:] all pixels
    img[:] = [b, g, r]  # Corrected order: (B, G, R)
    
    # Check for button press and break the loop if ESC is pressed
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

<h3>denoising</h3>
import cv2
import numpy as np
img=cv2.imread("butterfly.jpg")
dst=cv2.fastNlMeansDenoisingColored(img,None,10,10,7,15)
#second 10 is for filter strength of colour components 
#15 window size average weighted for a given pixel 
cv2.imshow("Original",img)
cv2.imshow("Denoised",dst)
<h3>histogram</h3>
import cv2
from matplotlib import pyplot as plt
image=cv2.imread("butterfly.jpg",1)
cv2.imshow("Original Image",image)
#[image]: This specifies the source image. Here, it's assumed to be a 
#single-channel grayscale image.
#[0]: This specifies the channel index (0 for grayscale).
#None: The mask used to calculate the histogram for the whole image.
#[256]: This specifies the number of bins for the histogram.
#[0, 256]: This specifies the range of pixel values.
histogram=cv2.calcHist([image],[0],None,[256],[0,256])
plt.plot(histogram,color="c")
plt.show()
<h3>face detection</h3>
import cv2

# Load cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Check if cascades are loaded successfully
if face_cascade.empty():
    raise IOError("Error: Unable to load face cascade classifier XML file")
if eye_cascade.empty():
    raise IOError("Error: Unable to load eye cascade classifier XML file")

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 127, 255), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

<h3>extracting images</h3>
import cv2
video=cv2.VideoCapture('Videopr.mp4')
currentframe=0 #frame index
while(True):
   retrn,frame=video.read() # true value in retrn and frame in frame
   if retrn:
       cv2.imwrite('img'+str(currentframe)+'.jpg',frame)
       currentframe+=1
   else:
       break
video.release()
cv2.destroyAllWindows()
<h3>reverse mode</h3>
import cv2

capture = cv2.VideoCapture("video_pr.mp4")

if capture.isOpened() is False:
    print("Error opening video")

frame_idx = capture.get(cv2.CAP_PROP_FRAME_COUNT)-1
print("Starting Frame: '{}'".format(frame_idx))

while capture.isOpened() and frame_idx >= 0:

     capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

     ret, frame = capture.read()
     if ret is True:

       cv2.imshow('Frame in Reverse', frame)

       frame_idx = frame_idx - 1
       print("Next index: '{}'".format(frame_idx))

       if cv2.waitKey(30) == ord('q'):
          break
     else:
        break

capture.release()
cv2.destroyAllWindows()
</pre></p>
</body>
</html>
