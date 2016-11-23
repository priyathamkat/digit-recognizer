# digit-recognizer

For [this](https://www.kaggle.com/c/digit-recognizer) Kaggle competition of the same name.

This method currently gives a score of `0.99129` on the public leaderboard. The network used is a simple CNN defined [here](https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html),
except the max pooling layers are replaced by gated pooling layers. You can read more about gated pooling in [this](https://arxiv.org/abs/1509.08985) paper.
