import os
import skimage.draw
import tensorflow as tf
from .base import rand_index, boundary_match, f_score
import numpy as np
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

ROOT_DIR = ''


class InferenceConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "postit"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.8


class MaskRCNN:
    def __init__(self):
        config = InferenceConfig()
        config.display()

        # Local path to trained weights file
        MY_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_postit.h5")

        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=os.path.join(ROOT_DIR, "logs"), config=config)

        # Load weights trained on MS-COCO
        model.load_weights(MY_MODEL_PATH, by_name=True)
        self.model = model

    def evaluate(self, image, ground_truth):
        segmentation = self.detect(image)
        rand = rand_index(segmentation, ground_truth)
        precision, recall = boundary_match(segmentation, ground_truth)
        evaluation = {"npri_score": rand,
                      "precision": precision,
                      "recall": recall,
                      "f_score": f_score(precision, recall)}
        return evaluation

    def detect(self, image):
        # load the input image and grab the image dimensions
        # Run detection
        results = self.model.detect([image], verbose=1)

        # Visualize results
        class_names = ['BG', 'postit']

        r = results[0]

        boxes = r['rois']
        masks = r['masks']

        N = r['rois'].shape[0]
        masked_image = np.zeros(image.shape[:2], dtype=np.uint32)

        for i in range(N):
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue

            #y1, x1, y2, x2 = boxes[i]

            # Mask
            mask = masks[:, :, i]
            masked_image[mask] = i

        return masked_image

