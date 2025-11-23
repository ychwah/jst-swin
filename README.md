# Source code for the Joint Structure-Texture model for image decomposition

You may find here the source code of the paper [Joint structure-texture low dimensional modeling for image decomposition with a plug and play framework.](https://www.hal.science/hal-04648963/file/article_pnp_siam.pdf)

The demo.py file may be run in order to produce a simple example of a decomposition with a single regularization function.


```
+--- src
|   contains the source code for the image decomposition using a single
|   regularization function R(u,v)
+--- saved_models
|   contains the weights of the neural networks
+--- model_generation
|   source code of the training procedure
+--- dataset
|   sample of the dataset used in the paper
    \--- test_dataset_synthetic.zip
           archived test dataset of synthetic images used in the paper.      
```