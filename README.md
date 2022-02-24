##  Car Classification
In this project, I built a model capable if correctly classifying cars as either Sedans, Coupes, SUVs or Trucks. Data used for this project is comprised of approximately 83,000 images personally scraped from the web ([see web_scraper_repo](https://github.com/Oreolorun/Web-Scraping)). Images were collected evenly between all 4 car classes so as to prevent class imbalance.

A convolutional neural network with a custom architecture inspired by the VGG group of architectures was used for model building. Two distinct models were built, one which works on 75x75 pixel images and another which works on 100x100 pixel images. The idea here is to create two models one which is trained to recognise more general features (75x75 pixels) and another which is trained to recognise a more detailed image (100x100 pixels).

Baselines are trained for each model with the baselines handling overfitting only via metric monitoring and optimisation by learning rate tuning. Final models however handled overfitting by batch normalization while optimisation was handled via learning rate tuning as well.

Upon optimization and analysis of test results, both the 100px and 75px batch normalized models showed similar performances (95.91% and 96.08% respectively) but their performance was quite different on individual car classes with the 100px model doing better in identifying Coupes and Trucks while the 75px model does a better job in identifying Sedans and SUVs. Both final models are then ensembled to take advantage of their unique strengths resulting in two ensembling modes with even better performances (96.17% and 97.18%).

Prototyping is done in the **CarTypeImageRecognition.ipynb** notebook file complete with comments, docstrings, visualisations, model explainability and a step-by-step walk thorough of the coding logic. Deployment is done via the python script **data_app.py**, culminating in a live [streamlit app](https://share.streamlit.io/oreolorun/image-recognition/main/data_app.py).

*if notebook file takes too long to render please open on [colab](https://colab.research.google.com/github/Oreolorun/Image-Recognition/blob/main/CarTypeImageRegcognition.ipynb).*

---

![SmartSelect_20220110-173716_Ipynb Viewer](https://user-images.githubusercontent.com/92114396/154482513-bdbb7e91-1c1c-41dd-af14-2c608ca60db5.jpg)
![shap2](https://user-images.githubusercontent.com/92114396/154482585-b56fc39f-3000-4fae-b64c-528dabc47296.png)
