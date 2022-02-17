##  Car Classification
In this project, I built a model capable if correctly classifying cars as either Sedans, Coupes, SUVs or Trucks. Data used for this project is comprised approximately 83,000 images personally scraped from the web ([see web scraper repo](https://github.com/Oreolorun/Web-Scraping)). Images were collected evenly between all 4 car classes so as to prevent class imbalance.

A convolutional neural network with a custom architecture inspired by the VGG group of architectures was used for model building. Two distinct models were built, one which workes in 75x75 pixel images and another which works on 100x100 pixel images. The idea here is to create two models one which is trained to recognise more general features (75x75 pixels) and another which is trained to recognise a more detailed image with more features (100x100 pixels).

Baselines are trained for each model with the baselines handling overfitting only via metric monitoring and optimisation by learning rate tuning. Final models however handled overfitting by batch normalization and optimisation by learning rate tuning as well.

Upon optimization and analysis of test results, both the 100px and 75px final models showed similar performances (95.91% and 96.08% respectively) but their performance was quite different on individual car classes with the 100px model doing better in identifying Coupes and Trucks while the 75px model does better in identifying Sedans and SUVs. Both final models are then ensembled to take advantage of their unique strengths resulting in two ensembling modes with even better performance (96.17% and 97.18%).

Prototyping is done in a notebook file complete with comments, docstrings, visualisations and a step-by-step walk thorough of the coding logic. Deployment is done via python script data_app.py with a live [streamlit app](https://share.streamlit.io/oreolorun/image-recognition/main/data_app.py)

---

![SmartSelect_20220110-173716_Ipynb Viewer](https://user-images.githubusercontent.com/92114396/154482513-bdbb7e91-1c1c-41dd-af14-2c608ca60db5.jpg)
![shap2](https://user-images.githubusercontent.com/92114396/154482585-b56fc39f-3000-4fae-b64c-528dabc47296.png)
