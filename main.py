import cv2
from random import randrange

img_file = 't.jpg'
video = cv2.VideoCapture('mn.mp4')


# pre traine data
trained_body_data = cv2.CascadeClassifier('haarcascade_fullbody.xml')
# create car classifier
car_tracker = cv2.CascadeClassifier('cars.xml')

while True:
        (read_successful,frame) = video.read()

        if read_successful:
                grey_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        else:
                 break

        # detect cars
        cars = car_tracker.detectMultiScale(grey_image)
        predestriants = trained_body_data.detectMultiScale(grey_image)
        for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)
        for (x, y, w, h) in predestriants:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("cars",frame)

        key =cv2.waitKey(1)
        # this for stops this when click the q button in ASCII Q==81 && q==113
        if key == 81 or key == 113:
                break

video.release()


""""
#create opencv image
img = cv2.imread(img_file)
#black and white
black = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#create car classifier
car_tracker = cv2.CascadeClassifier(classifier)
#detect cars
cars = car_tracker.detectMultiScale(black)
print(cars)
for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)


cv2.imshow("cars",img)

cv2.waitKey()
"""


# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
