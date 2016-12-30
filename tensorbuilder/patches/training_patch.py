from tflearn.helpers.trainer import TrainOp, Trainer
from tensorbuilder import TensorBuilder


TensorBuilder.Register(TrainOp, "tflearn.")
TensorBuilder.Register(Trainer, "tflearn.")

