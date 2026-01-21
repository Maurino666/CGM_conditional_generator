class Callback:
    """
    Base callback class. All methods are optional.
    """

    def on_train_start(self, trainer, model): pass

    def on_train_end(self, trainer, model): pass

    def on_epoch_start(self, trainer, model, epoch): pass

    def on_epoch_end(self, trainer, model, epoch, metrics): pass