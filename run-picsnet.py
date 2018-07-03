from process_image import ProcessImage

from picsnet import NeuralTransfer

content_path = ""

style_path = ""

height = 200

width = 200

style = ProcessImage(style_path, height, width)

content = ProcessImage(content_path, height, width)

style.process_image()

content.process_image()

trans = NeuralTransfer(style.processed_image, content.processed_image, height, width, 1, 0.00005, 0.00001)

trans.run_stylizing()
