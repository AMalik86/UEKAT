import cv2
from flask import Flask, request
from flask_restful import Resource, Api

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

app = Flask(__name__)
api = Api(app)
class PeopleCounterStatic(Resource):
    def get(self):
        # load image
        image = cv2.imread('12.jpg')
        image = cv2.resize(image, (700, 400))

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # show the output images
        cv2.imshow("People detector", image)
        cv2.waitKey(0)

        return {'peopleCount': len(rects)}



api.add_resource(PeopleCounterStatic, '/')

if __name__ == '__main__':
    app.run(debug=True)
