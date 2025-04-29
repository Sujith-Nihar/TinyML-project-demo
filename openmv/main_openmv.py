import sensor, time, tf

# Initialize
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)

# Load model
net = tf.load('tiny_model_quantized.tflite', load_to_fb=True)

while(True):
    img = sensor.snapshot()
    img_small = img.copy().resize(10, 1)  # Resize to 10 pixels wide

    input_data = []
    for i in range(10):
        input_data.append(img_small.get_pixel(i, 0)[0] / 255.0)

    output = net.classify(input_data)

    if output[0][0] > output[0][1]:
        print("Dark Image")
    else:
        print("Bright Image")

    time.sleep(500)
