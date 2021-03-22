import comet_ml
import getpass, os
os.environ["COMET_API_KEY"] = getpass.getpass("Paste your COMET API KEY: ")

from tensorflow import keras

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class ConfusionMatrixCallback(keras.callbacks.Callback):
    def __init__(self, experiment, inputs, targets, interval):
        self.experiment = experiment

        self.inputs = inputs
        self.targets = targets
        self.interval = interval

    def index_to_example(self, index):
        image_array = self.inputs[index]
        image_name = "confusion-matrix-%05d.png" % index
        results = experiment.log_image(image_array, name=image_name)
        # Return sample, assetId (index is added automatically)
        return {"sample": image_name, "assetId": results["imageId"]}

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.interval != 0:
            return

        predicted = self.model.predict(self.inputs)
        self.experiment.log_confusion_matrix(
            self.targets,
            predicted,
            labels=LABELS,
            index_to_example_function=self.index_to_example,
            title="Confusion Matrix, Epoch #%d" % (epoch + 1),
            file_name="confusion-matrix-%03d.json" % (epoch + 1),
        )


# Model configuration
batch_size = 8092
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = categorical_crossentropy
no_classes = 10
no_epochs = 5
optimizer = Adam()
verbosity = 1
validation_split = 0.2
interval = 1

# Load CIFAR-100 data
(input_train, target_train), (input_test, target_test) = cifar10.load_data()

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

target_train, target_test = tuple(
    map(lambda x: keras.utils.to_categorical(x), [target_train, target_test])
)

# Parse numbers as floats
input_train = input_train.astype("float32")
input_test = input_test.astype("float32")

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

# Create the model
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(no_classes, activation="softmax"))

# Compile the model
model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])

PROJECT_NAME = "p100-example"
experiment = comet_ml.Experiment(
    project_name=PROJECT_NAME,
    auto_metric_logging=True,
    auto_param_logging=True,
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,
)
experiment.log_parameter("batch_size", batch_size)
confmat = ConfusionMatrixCallback(experiment, input_test, target_test, interval)

# Fit data to model
model.fit(
    input_train,
    target_train,
    batch_size=batch_size,
    epochs=no_epochs,
    verbose=verbosity,
    validation_split=validation_split,
    callbacks=[confmat],
)

print(experiment.display())
print(experiment.end())
