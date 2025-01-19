class BaseLayer:
    def __init__(self):
        """
        Base class for all layers.
        """
        self.testing_phase = False # whether layer is in testing phase
        self.trainable = False # whether the layer has trainable parameters