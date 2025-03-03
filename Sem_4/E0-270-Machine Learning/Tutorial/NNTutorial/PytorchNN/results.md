# Results

| Experiment no | Dataset       | Train Accuracy | Test Accuracy | Remarks                        |
|---------------|---------------|----------------|---------------|--------------------------------|
| 1             | MNIST         | 0.95           | 0.95          | MLP (80K Params)               |
| 2             | Fashion MNIST | 0.87           | 0.85          | MLP                            |
| 3             | Fashion-MNIST | 0.88           | 0.87          | SimpleConv (9K Params)         |
| 4             | Fashion-MNIST | 0.90           | 0.89          | Conv (1M Params)               |
| 5             | Fashion-MNIST | 0.99           | 0.92          | Conv + Adam Optimizer          |
| 6             | Cifar10       | 0.75           | 0.66          | Conv (33K Params)              |
| 7             | Cifar10       | 0.87           | 0.66          | Conv + batchnorm               |
| 8             | Cifar10       | 0.99           | 0.76          | Conv + batchnorm (0.6M Params) |
| 9             | Cifar10       | 0.95           | 0.81          | Conv + batchnorm + dropout     |
