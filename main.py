
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

        return {'peopleCount': len(rects)}


class PeopleCounterDynamicUrl(Resource):

    @app.route('/dynamic', methods=['GET'])
    def get_query_string():
        return request.query_string

        # TODO:
        # 1. Pobrać zdjęcie z otrzymanego adresu
        # 2. Pobrane zdjęcie można zapisać na dysku lub przetwarzać je w pamięci podręcznej
        # 3. Załadowane zjęcie do zmiennej image przekazać do algorytmu hog.detectMultiScale i zwrócić z endpointu liczbę wykrytych osób.

    def download_file(url, dst_path):
        try:
            with urllib.request.urlopen(url) as web_file:
                data = web_file.read()
                with open(dst_path, mode='wb') as local_file:
                    local_file.write(data)
        except urllib.error.URLError as e:
            print(e)
        url = request.args.get('/dynamic')
        dst_path = 'zdjecie.jpg'

        image = download_file(url, dst_path)
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        return {'peopleCount': len(rects)}





api.add_resource(PeopleCounterStatic, '/')
api.add_resource(PeopleCounterDynamicUrl, '/dynamic')

if __name__ == '__main__':
    app.run(debug=True)
