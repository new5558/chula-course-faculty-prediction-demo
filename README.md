# chula-course-faculty-prediction-demo

## [Online Demo](https://share.streamlit.io/new5558/chula-course-faculty-prediction-demo/src/main.py)
![image](https://user-images.githubusercontent.com/12471844/144892811-52887bcb-3c1f-4bc7-bb40-5c8081cb2cc1.png)

## Ideas
We want to predict faculty in Chulalongkorn university based on courses Thai description. We may use this model to extract features from course descrition that may be developed further into course Recommender system. 

Features:
- Interactive web demo by Streamlit
- Map result to faculty name
- Interpret model with SHAP

## Prerequisite
- Need [git LFS](https://git-lfs.github.com/) for cloning this repo.
- Need Docker to be installed 

## Dependencies
- Finetuned WangchanBERTA Model from [Chula-course-recommender-proof-of-concept](https://github.com/new5558/Chula-course-recommender-proof-of-concept)
- Faculty JSON dictionary from [DSC Chula](https://raw.githubusercontent.com/DSCChula/chula-util/main/src/faculties.json)

## Development
Run on docker compose:    
`docker-compose -f docker-compose.dev.yml up streamlit --build`

## Heroku deployment
Not working on free version of Heroku becuase not of memory error