
from process_image import ProcessImage

from picsnet import NeuralTransfer

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--content", "-c", help="Path to content image")

parser.add_argument("--style", "-s", help="Path to style image")

parser.add_argument("--output_height", "-oh", help="Output image height", type=int)

parser.add_argument("--output_width", "-ow", help="Output image width", type=int)

parser.add_argument("--retain_color", "-rc", help="Retain content image's color (boolean True or False)", type=bool)

parser.add_argument("--content_weight", "-cw", help="Value of content weight", type=float)

parser.add_argument("--style_weight", "-sw", help="Value of style weight", type=float)

parser.add_argument("--total_variation_weight", "-tw", help="Value of total variation weight", type=float)

parser.add_argument("--learning_rate", "-lr", help="Value of learning rate", type=float)

parser.add_argument("--num_iterations", "-i", help="Number of iterations", type=int)

args = parser.parse_args()


content_path = args.content

style_path = args.style

height = args.output_height

width = args.output_width

retain_content_color = args.retain_color

content = ProcessImage(content_path, height, width)

style = ProcessImage(style_path, height, width)


if retain_content_color:

    style.image = ProcessImage.histogram_match(content.image, style.image)


style.process_image()

content.process_image()

trans = NeuralTransfer(style.processed_image, content.processed_image, height, width, args.style_weight,
                       args.content_weight, args.total_variation_weight, args.learning_rate, args.num_iterations)

trans.run_stylizing()


