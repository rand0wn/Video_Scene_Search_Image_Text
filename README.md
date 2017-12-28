# Video_Scene_Search_Image_Text(Show and Tell Model CNN-RNN)  

Search around your own videos, navigate through scenes from image and text queries.  

## Video Scene Search Using Image and Text  
* Train your own videos to search  
* Search Videos and Video Scene Using Image  
* Search Videos and Video Scene Using Text  
* Search through multiple videos  
* Easily work through REST API  

#### Dependencies
* [Python 2.7](https://www.python.org)  
* [Tensorflow > 1.2](https://www.tensorflow.org/)  
* [nltk](https://pypi.python.org/pypi/nltk)  
* [flask](http://flask.pocoo.org/)  

## Pretrained Model
* Download Link: https://www.dropbox.com/s/9xmjkm8phlx64ek/vs_model.zip?dl=0  
  
## Model Information  
* [Show and Tell Model](https://research.googleblog.com/2016/09/show-and-tell-image-captioning-open.html)  
* Encoder-Decoder Neural Network  
* Encoder - Inception v3 image recognition model pretrained on the ILSVRC-2012-CLS image classification dataset.  
* Decoder - LSTM Network Trained on Captions represented with an embedding model  
* Beam Search as Caption Generator  

## Instructions  
* Download Pretrained Model and extract in the models folder    
* Run REST API with 'video_rest.py'    

### Training your Videos
* Train your video directly from the REST API  
* Example: http://localhost:5003/train/home/Downloads/RDG.webm/, Video Path is: home/Downloads/RDG.webm/, don't miss the '/'   
* You can train multiple videos, data is stored in vd_data folder  

### Search from Videos
* You can search scenes around multiple videos from Image or Text  
* Image: http://localhost:5003/isearch/home/imgs/1346353449_eminem.jpg/, Image Path: /home/imgs/1346353449_eminem.jpg/   
* Text: http://localhost:5003/tsearch/group%20people%20wine%20glass   


