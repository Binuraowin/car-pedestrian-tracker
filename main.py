import cv2
from random import randrange

img_file = 't.jpg'
#create opencv image
img = cv2.imread(img_file)
#black and white
black = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


classifier = 'cars.xml'
#create car classifier
car_tracker = cv2.CascadeClassifier(classifier)
#detect cars
cars = car_tracker.detectMultiScale(black)
print(cars)
for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)


cv2.imshow("cars",img)

cv2.waitKey()

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
