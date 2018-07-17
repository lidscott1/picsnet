
import tensorflow as tf
import numpy as np
from process_image import ProcessImage


class NeuralTransfer(object):

    def __init__(self, style_image,
                 content_image,
                 height,
                 width,
                 style_weight,
                 content_weight,
                 tv_weight,
                 learning_rate,
                 iterations = 100,
                 blank_image=False):

        self.style_image = style_image

        self.content_image = content_image

        self.height = height

        self.width = width

        self.style_weight = style_weight

        self.content_weight = content_weight

        self.tv_weight = tv_weight

        self.learning_rate = learning_rate

        self.iterations = iterations

        if blank_image:

            self.tinker_image = np.random.uniform(0, 255, (1, self.height, self.width, 3))

        else:

            self.tinker_image = content_image

        self.layers_for_opt = ["block1_conv1", "block2_conv1", "block2_conv2", "block3_conv1",
                               "block3_conv2", "block4_conv1", "block4_conv3", "block5_conv1", "block5_conv3"]

    @staticmethod
    def gram_matrix(model, layer):

        dimension = model.get_layer(layer).output.shape[3].value

        permute = tf.transpose(model.get_layer(layer).output[0], perm=[2, 0, 1])

        reshaped = tf.reshape(permute, [dimension, -1])

        final = tf.matmul(reshaped, reshaped, transpose_b=True)

        return final

    def content_loss(self, input_image, content_image):

        increment_loss = tf.reduce_sum(tf.square(input_image - content_image))/(self.height*self.width)

        return increment_loss


    def style_loss(self, input_gramm, style_gramm, weight):

        increment_loss = tf.reduce_sum((tf.square(input_gramm - style_gramm)))

        increment_loss = weight * increment_loss

        return increment_loss

    def tv_loss(self, image):

        increment_loss = (tf.reduce_sum(tf.abs(image[:, 1:, :, :] - image[:, :-1, :, :])) +

                          tf.reduce_sum(tf.abs(image[:, :, 1:, :] - image[:, :, :-1, :])))

        return increment_loss

    def __make_style_grams__(self):

        sess1 = tf.InteractiveSession()

        style_tensor = tf.constant(self.style_image)

        model = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=style_tensor)

        model.trainable = False

        self.grams = {}

        for layer in self.layers_for_opt:

            self.grams[layer] = sess1.run(self.gram_matrix(model, layer))

        sess1.close()

        tf.reset_default_graph()


    def __stylize__(self):

        sess2 = tf.InteractiveSession()

        tinkered_tensor = tf.Variable(self.content_image, trainable=True)

        content_tensor = tf.constant(self.content_image)

        model2 = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=tinkered_tensor)

        model2.trainable = False

        model3 = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=content_tensor)

        model3.trainable = False

        loss = tf.Variable(initial_value=tf.zeros(1))

        loss.initializer.run()

        c_loss = self.content_loss(model3.get_layer("block1_conv2").output, model2.get_layer("block1_conv2").output)

        t_loss = self.tv_loss(tinkered_tensor)

        s_loss = 0

        for layer in self.layers_for_opt:

            current_gram = self.gram_matrix(model2, layer)

            s_loss += self.style_loss(self.grams[layer], current_gram, 1)


        loss = self.content_weight * c_loss + self.style_weight * s_loss + self.tv_weight * t_loss

        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        run_it = opt.minimize(loss, var_list=tinkered_tensor)

        tinkered_tensor.initializer.run()

        sess2.run(tf.variables_initializer(opt.variables()))

        for i in range(self.iterations):

            _, current_loss = sess2.run([run_it, loss])

            print(i, current_loss)

            if i != 0 and i % 10 == 0:

                final_image = sess2.run(tinkered_tensor)[0]

                img = ProcessImage.unprocess_image(final_image)

                img.save("temp_output.jpg")

                img.show()

                continue_optimizing = input("Continue optimizing (yes / no)? ")

                if continue_optimizing == "no":

                    break

                else:

                    continue

        self.final_image = sess2.run(tinkered_tensor)[0]

        sess2.close()

    def run_stylizing(self):

        self.__make_style_grams__()

        self.__stylize__()

