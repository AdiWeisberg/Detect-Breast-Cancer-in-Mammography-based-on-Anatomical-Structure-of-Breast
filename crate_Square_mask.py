
import cv2

img =cv2.imread('mask.png',cv2.THRESH_BINARY)
img_full = cv2.imread('mask_full_image.png')

# masked = cv2.bitwise_and(img_full, img)
cnts , _= cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(cnts)
countors = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
x,y,w,h = cv2.boundingRect(countors[0])
print("point:",x,",",y)
print(w,",",h)
roi=img[y:y+h,x:x+w]
new_image = cv2.rectangle(img_full,(x,y),(x+w,y+h),(200,0,0),10)


cv2.namedWindow('output1', cv2.WINDOW_NORMAL)
cv2.resizeWindow('output1', 600, 600)
cv2.imshow("output1", new_image)
cv2.waitKey(0)
