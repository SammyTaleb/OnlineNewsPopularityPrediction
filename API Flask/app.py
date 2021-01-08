# -*- coding: utf-8 -*-
from flask import Flask,render_template,request
import pickle
import numpy as np
import json

app = Flask(__name__, template_folder='templates')
clf_rf = pickle.load(open('clf_rf.pkl', 'rb'))
clf_gb = pickle.load(open('clf_gb.pkl', 'rb'))

features1=['n_tokens_title', 'n_tokens_content', 'n_non_stop_unique_tokens','num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos','average_token_length', 'num_keywords']
features2=['data_channel_is_lifestyle','data_channel_is_entertainment', 'data_channel_is_bus','data_channel_is_socmed', 'data_channel_is_tech','data_channel_is_world']
features3=['kw_min_min', 'kw_max_min', 'kw_min_max','kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg','self_reference_min_shares', 'self_reference_max_shares','self_reference_avg_sharess']
features4=['weekday_is_monday', 'weekday_is_tuesday','weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday','weekday_is_saturday', 'weekday_is_sunday', 'is_weekend']
features5=['LDA_00','LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 'global_subjectivity','global_sentiment_polarity', 'global_rate_positive_words','global_rate_negative_words', 'rate_positive_words','rate_negative_words', 'avg_positive_polarity', 'min_positive_polarity','max_positive_polarity', 'avg_negative_polarity','min_negative_polarity', 'max_negative_polarity', 'title_subjectivity','title_sentiment_polarity', 'abs_title_subjectivity','abs_title_sentiment_polarity']
default_values1=[11.0,1368.0,0.736280486682,10.0,0.0,1.0,0.0,4.34502923977,7.0]
default_values2=[0.0,0.0,1.0,0.0,0.0,0.0]
default_values3=[-1.0,1500.0,0.0,843300.0,243028.571429,0.0,3840.0,2759.49477255,0.0,0.0,0.0]
default_values4=[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
default_values5=[0.7615147787450001, 0.0285793249236, 0.152753325375, 0.028578074453500003, 0.0285744965031, 0.508952206797, 0.147219734289, 0.0533625730994, 0.0204678362573, 0.722772277228, 0.277227722772, 0.352910810651, 0.0333333333333, 1.0, -0.303571428571, -0.7, -0.05, 0.33333333333299997, 0.166666666667, 0.166666666667, 0.166666666667]
params={'len1':len(features1),'len2':len(features2),'len3':len(features3),'len4':len(features4),'len5':len(features5),
        'features1':features1,'features2':features2,'features3':features3,'features4':features4,'features5':features5,
        'def1':default_values1,'def2':default_values2,'def3':default_values3,'def4':default_values4,'def5':default_values5}


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/notebook")
def notebook():
    return render_template("JupyterNotebook.html",title='Dataset Exploration and Models Building Notebook')

@app.route("/randomForest")
def randomForest():
    return render_template("randomForest.html",title='Random Forest Prediction',**params)



@app.route('/randomForest/prediction',methods=['POST'])
def predictRF():
    features=[x for x in request.form.values()]
    default=[*default_values1,*default_values2,*default_values3,*default_values4,*default_values5]
    for i in range(len(features)):
        if features[i]=='':
            print(i)
            print(default[i])
            features[i]=default[i]
    features=[float(x) for x in features]  
    X = [np.array(features)]
    print(X)
    Y = clf_rf.predict(X)
    if Y==1:
    	output = 'Popular'
    elif Y==0:
   		output = 'Unpopular'
    else:
   		output = 'Error'
    return render_template("randomForest.html",title='Random Forest Prediction',
                           prediction_text='The article should be {}'.format(output),**params)

@app.route("/gradientBoosting")
def gradientBoosting():
    return render_template("gradientBoosting.html",title='Gradient Boosting Prediction',**params)

@app.route('/gradientBoosting/prediction',methods=['POST'])
def predictGB():
    features=[x for x in request.form.values()]
    default=[*default_values1,*default_values2,*default_values3,*default_values4,*default_values5]
    for i in range(len(features)):
        if features[i]=='':
            print(i)
            print(default[i])
            features[i]=default[i]
    features=[float(x) for x in features]  
    X = [np.array(features)]
    print(X)
    Y = clf_gb.predict(X)
    if Y==1:
    	output = 'Popular'
    elif Y==0:
   		output = 'Unpopular'
    else:
   		output = 'Error'

    return render_template("gradientBoosting.html",title='Gradient Boosting Prediction',
                           prediction_text='The article should be {}'.format(output),**params)



if __name__ == "__main__":
    app.run(debug=True)