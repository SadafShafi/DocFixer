# DocFixer

**Abstract**

Document deskewing is a fundamental problem in document image processing. While existing methods have limitations, such as Hough Line Transformation that can deskew images upside down, and Deep Learning models that require huge amounts of human labour and computational resources and still fail to deskew while taking care of orientation, OCR-based methods also struggle to read text when it is tilted. In this paper, we propose a novel, simple, cost-effective deep learning method for fixing the skew and orientation of documents. Our approach reduces the search space for the machine learning model to predict whether an image is upside down or not, avoiding the huge search space of predicting an angle between 0 and 360. We finetune a MobileNetV2 model, which was pre-trained on imagenet, using only 200 images and achieve good results. This method is useful for automation-based tasks, such as data extraction using OCR technology, and can greatly reduce manual labour. We provide the source code for this project on GitHub at https://github.com/SadafShafi/DocFixer

* Run `Doc_Fixer_Training_the_model_(inversion_rectifier_training).ipynb` on jupyter notebook or google colab to train the Inversion Rectifier 
* Once the model is trained, get that trained model and its architecture (as json) in directory Model_100L_1.0_acc
* Now add paths of the model and its architecture in `model_path` and `model_architecture` in `testing_on_individual_docs__orginal.py` and you can use the model rightaway in production 
* In order to test the whole working of project import the model and its architecture into `Testing_the_trained_model_.ipynb`.
