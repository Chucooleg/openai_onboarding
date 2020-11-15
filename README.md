[OpenAI Scholar Blog](https://wordpress.com/view/bitbybitworks.wordpress.com)

Logistic Regression
- custom autograd functions (manual forward and backward)
- custom simple SGD optimizer
- dataset. dataloader
- cpu only

Cuda Experiments
- memory monitoring
- synchronization
- TODO: profiling tools

Simple MLP
- GPU training
- Distributed DataParallel training
- Readings to understand distributed model training paradigms

Transformer Base
- code from scratch
- pytorch lightning integration
- experiments to examine impact of: WQKVO initialization, embedding initialization, learning rate schedule, untying initial weights of encoder-decoder stacks.
- toy dataset: copy symbols, reverse copied symbols, +1 numer series, sum of last two input numbers
- midsize dataset: human to machine dates translation, polynomial expansion
- full dataset: WMT 2014 English-German (Work in progress)
- 16-bit training (Work in progress)
