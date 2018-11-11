
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, redirect, render_template, request, url_for
import json
import pandas as pd
import os
import numpy as np
import scipy
from scipy.io import mmread
import surprise
from surprise import Reader, Dataset, SVD, evaluate


df = pd.read_csv("/home/zh2290/mysite/static/data/df_name.csv")
cosine_sim=mmread("/home/zh2290/mysite/static/data/cosine_sim.mtx")
df_ratings = pd.read_csv("/home/zh2290/mysite/static/data/df_ratings.csv")
df_ratings= df_ratings[['business_id',"user_id","stars"]]
category = pd.read_csv("/home/zh2290/mysite/static/data/df_features.csv")

app = Flask(__name__)

def get_svd(df_ratings):
    reader =Reader()
#training
    data = Dataset.load_from_df(df_ratings, reader)
    data.split(n_folds=5)
    svd = SVD()
    trainset = data.build_full_trainset()
    svd.fit(trainset)
    return svd


def get_recommendations(title):
    #df.iloc[np.where(df['name'] == title)[0][0],:]['business_id']
    #smd = df_tf.reset_index()
    #titles = smd['business_id']
    title=df[df['name'] == title]['business_id'].unique()[0]
    titles=df['business_id'].unique()

    indices = pd.Series(df.index[:len(titles)], index=titles)
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:51]
    restaurant_indices = [i[0] for i in sim_scores]
    result=titles[restaurant_indices]
    result1=df[df['business_id'].isin(result)]['name'].unique().tolist()
    return result1

def get_recommendation_account(account):
    svd=get_svd(df_ratings)
    scores=np.zeros(len(df['business_id'].unique()))
    for i in range(len(scores)):
        scores[i]= svd.predict(account,category['business_id'][i]).est
    result= pd.DataFrame(np.hstack((pd.DataFrame(df['business_id'].unique()),    pd.DataFrame(scores))))
    result.columns = ['business_id','scores']
#    result1 = result.sort('scores',ascending=False)
    result1=df[df['business_id'].isin(result['business_id'])]['name'].unique().tolist()
    return result1


def hybrid(userId, title):
    svd=get_svd(df_ratings)
    businessid = df.iloc[np.where(df['name'] == title)[0],:]["business_id"].unique()[0]
    idx=np.where(pd.DataFrame(list(enumerate(df['business_id'].unique())))[1] == businessid)[0]

    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    restaurant_indices = [i[0] for i in sim_scores]
    restaurant_id = df['business_id'].unique()[restaurant_indices]
    df[df['business_id'].isin(restaurant_id)]['name'].unique().tolist()

    scores= np.zeros(restaurant_id.shape[0])
    for i in range(restaurant_id.shape[0]):
        scores[i]= svd.predict(userId,restaurant_id[i]).est
    result= pd.DataFrame(np.hstack((pd.DataFrame(restaurant_id),    pd.DataFrame(scores))))
    result.columns = ['business_id','scores']

    final=result.sort_values('scores',ascending=False)

    return(df[df['business_id'].isin(final["business_id"])]['name'].unique().tolist())





#category = pd.read_csv('../ .... ',index_col=None)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("main_page.html")





@app.route('/user_detail', methods=["POST"])

def third():
    data={}
    if request.form:
        form_data = request.form
        data['form'] = form_data
        user_city = data['form']['city']
        user_food = data['form']['food']
        user_option = data['form']['option']
        user_special = data['form']['specialInput']
        user_account = data['form']['account']
        filter1 = category[category[user_food] == 1].iloc[:,14]
        filter1_name=df[df['business_id'].isin(filter1)]['name'].unique()
        filter2 = category[category[user_option] == 1].iloc[:,14]
        filter2_name = df[df['business_id'].isin(filter2)]['name'].unique()
        filter3 = set(filter1) | set(filter2)
        filter3_name= set(filter1_name) | set(filter2_name)



        if data['form']['specialInput'] =='' and data['form']['account']=='':
            res = pd.unique(filter3_name)
        elif data['form']['specialInput']=='' and data['form']['account']!='':
            res = get_recommendation_account(data['form']['account'])
            a=np.where(pd.DataFrame(res).isin(filter3_name))[0]
            res=np.array(res)[a]
        elif data['form']['specialInput']!='' and data['form']['account']=='':
            res =get_recommendations(data['form']['specialInput'])
            a=np.where(pd.DataFrame(res).isin(filter3_name))[0]
            res=np.array(res)[a]
        elif data['form']['specialInput']!='' and data['form']['account']!='':
            res= hybrid(data['form']['account'],data['form']['specialInput'])
            a=np.where(pd.DataFrame(res).isin(filter3_name))[0]
            res=np.array(res)[a]
    return render_template('user_detail.html',city=user_city,food=user_food,option=user_option,restaurant=user_special,e=user_account,recommend=res)





