import numpy as np
import os
from keras.models import load_model
from scripts.metrics import dice_coefficient, weighted_categorical_crossentropy

# Load test images
test_imgs = np.load('/.../')
test_labels = np.load('/.../')

# Load trained model
model = load_model('/.../', custom_objects={"dice_coefficient": dice_coefficient})

# Evaluate model
score = model.evaluate(test_imgs, test_labels, verbose=1, batch_size=2)

# Make predictions on test images
predicted_masks_cnn = model.predict(test_imgs, batch_size=2, verbose=1)

# Save predictions
np.save(os.path.join('/.../'), predicted_masks_cnn)

# Print results
print('Test loss:', score[0])
print('Test accuracy:', score[1])