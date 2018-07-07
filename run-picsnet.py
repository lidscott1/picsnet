from process_image import ProcessImage

from picsnet import NeuralTransfer

content_path = "/Users/liam/Desktop/Projects/picsnet/photos/mich.jpg"

style_path = "/Users/liam/Desktop/Projects/picsnet/photos/first_star.jpg"

height = 1600

width = 550

retain_content_color = True

content = ProcessImage(content_path, height, width)

style = ProcessImage(style_path, height, width)


if retain_content_color:

    style.image = ProcessImage.histogram_match(content.image, style.image)


style.process_image()

content.process_image()

trans = NeuralTransfer(style.processed_image, content.processed_image, height, width, 0.0005, 0.000005, 0.00001, 5)

trans.run_stylizing()
