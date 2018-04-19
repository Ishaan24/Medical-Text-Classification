# Medical-Text-Classification
Develop predictive models that can determine, given a medical abstract, which of 5 classes it falls in.
Medical abstracts describe the current conditions of a patient. 
Doctors routinely scan dozens or hundreds of abstracts each day as they do their rounds in a hospital and must quickly pick up on the salient information pointing to the patient’s malady.
You are trying to design assistive technology that can identify, with high precision, the class of problems described in the abstract.
In the given dataset, abstracts from 5 different conditions have been included: digestive system diseases, cardiovascular diseases, neoplasms, nervous system diseases, and general pathological conditions.


The goal of this competition is to allow you to develop predictive models that can determine, given a particular medical abstract, which one of 5 classes it belongs to.
As such, the goal would be to develop the best classification model.
As we have learned in class, there are many ways to represent text as sparse vectors. 
Feel free to use any of the code in activities or write your own for the text processing step.


The training dataset consists of 14442 records and the test dataset consists of 14438 records. 
We provide you with the training class labels and the test labels are held out. 
The data are provided as text in train.dat and test.dat, which should be processed appropriately.
train.dat: Training set (class label, followed by a tab separating character and the text of the medical abstract).
test.dat: Testing set (text of medical abstracts in lines, no class label provided).
