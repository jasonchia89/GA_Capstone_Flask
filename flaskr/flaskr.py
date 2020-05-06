# all the imports
#import os
#import sqlite3
import pandas as pd
import numpy as np
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml import Pipeline as PL
from pyspark.sql.functions import col
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import re
from flask import Flask, request, render_template

app = Flask(__name__) # create the application instance :)
#app.config.from_object(__name__) # load config from this file , flaskr.py

outlets = pd.read_csv("outlets.csv")
outlet_list = outlets.shop.tolist()
url_front = 'https://www.yelp.com/biz/'
outlet_url = [url_front+x for x in outlet_list]
url_alias = list(zip(outlet_url,outlet_list))

# Load default config and override config from an environment variable
#app.config.update(dict(
#    DATABASE=os.path.join(app.root_path, 'flaskr.db'),
#    SECRET_KEY='development key',
#    USERNAME='admin',
#    PASSWORD='default'
#))
#app.config.from_envvar('FLASKR_SETTINGS', silent=True)

#def connect_db():
#    """Connects to the specific database."""
#    rv = sqlite3.connect(app.config['DATABASE'])
#    rv.row_factory = sqlite3.Row
#    return rv

#def init_db():
#    db = get_db()
#    with app.open_resource('schema.sql', mode='r') as f:
#        db.cursor().executescript(f.read())
#    db.commit()

#@app.cli.command('initdb')
#def initdb_command():
#    """Initializes the database."""
#    init_db()
#    print('Initialized the database.')



#def get_db():
#    """Opens a new database connection if there is none yet for the
#    current application context.
#    """
#    if not hasattr(g, 'sqlite_db'):
#        g.sqlite_db = connect_db()
#    return g.sqlite_db

#@app.teardown_appcontext
#def close_db(error):
#    """Closes the database again at the end of the request."""
#    if hasattr(g, 'sqlite_db'):
#        g.sqlite_db.close()

@app.route('/')

def initial_ratings():
    return render_template('initial_ratings.html', list=outlet_list, url_alias=url_alias)

#def show_entries():
#    db = get_db()
#    cur = db.execute('select title, text from entries order by id desc')
#    entries = cur.fetchall()
#    return render_template('show_entries.html', entries=entries)

@app.route('/', methods=["GET", "POST"])

def getvalues_and_recommend():
    userid = 2552
    shop1 = request.form['shop1']
    rate1 = float(request.form['rate1'])
    shop2 = request.form['shop2']
    rate2 = float(request.form['rate2'])
    shop3 = request.form['shop3']
    rate3 = float(request.form['rate3'])
    shop4 = request.form['shop4']
    rate4 = float(request.form['rate4'])
    shop5 = request.form['shop5']
    rate5 = float(request.form['rate5'])
    shop6 = request.form['shop6']
    rate6 = float(request.form['rate6'])
    shop7 = request.form['shop7']
    rate7 = float(request.form['rate7'])
    shop8 = request.form['shop8']
    rate8 = float(request.form['rate8'])
    shop9 = request.form['shop9']
    rate9 = float(request.form['rate9'])
    shop10 = request.form['shop10']
    rate10 = float(request.form['rate10'])

    #creating a new spark session
    newspark = SparkSession.builder.appName('hybrid_rec').getOrCreate()
    #reading in prepped dataset for model-based collaborative filtering recommendation
    mbcf = newspark.read.csv('mbcf.csv', header=True, inferSchema=True)
    #making a copy for each new user input
    mbcf_try = mbcf
    vals = [(shop1,rate1,userid),(shop2,rate2,userid),(shop3,rate3,userid),(shop4,rate4,userid),(shop5,rate5,userid),(shop6,rate6,userid),(shop7,rate7,userid),(shop8,rate8,userid),(shop9,rate9,userid),(shop10,rate10,userid)]
    #pyspark's convention to adding new rows to the end of an existing spark dataframe-1
    newRows = newspark.createDataFrame(vals,mbcf_try.columns)
    #pyspark's convention to adding new rows to the end of an existing spark dataframe-2
    mbcf_try = mbcf_try.union(newRows)
    #converting df to pandas df for easier manipulation later on...
    mbcf_try_pd = mbcf_try.toPandas()
    #getting a look again at the outlets and ratings provided by userid2552 so we know which outlets to exclude in recommending outlets to userid2552 later on...
    user_item_2552 = mbcf_try_pd[mbcf_try_pd['userids']==2552]
    #as part of ALS requirements for the feature columns to be in numerical format, am converting both shops and userids to the double precision format just in case (even though userids is already in a float format)
    indexer_try = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in list(set(mbcf_try.columns)-set(['ratings']))]
    pipeline_try = PL(stages=indexer_try)
    transformed_try = pipeline_try.fit(mbcf_try).transform(mbcf_try)
    #rank=300 and regParam=0.1 was a pair of tuned best params while retuning als with train test split stratified for userids...
    als = ALS(rank=300, regParam=0.1, maxIter=20, seed=42, userCol='userids_index',itemCol='shops_index', ratingCol='ratings',coldStartStrategy='drop')
    #training the dataset containing the new user's ratings...
    als_model_rec = als.fit(transformed_try)
    #making recommendations for model-based collaborative filtering alone first, passing in all 981 outlets so as to ensure as much overlap between collaborative filtering and content-based filtering in the outlets that they generate rating predictions for
    recs=als_model_rec.recommendForAllUsers(981).toPandas()
    nrecs=recs.recommendations.apply(pd.Series) \
                .merge(recs, right_index = True, left_index = True) \
                .drop(["recommendations"], axis = 1) \
                .melt(id_vars = ['userids_index'], value_name = "recommendation") \
                .drop("variable", axis = 1) \
                .dropna()
    nrecs=nrecs.sort_values('userids_index')
    nrecs=pd.concat([nrecs['recommendation'].apply(pd.Series), nrecs['userids_index']], axis = 1)
    nrecs.columns = [

            'Shop_index',
            'Rating',
            'UserID_index'

         ]
    md=transformed_try.select(transformed_try['userids'],transformed_try['userids_index'],transformed_try['shops'],transformed_try['shops_index'])
    md=md.toPandas()
    dict1=dict(zip(md['userids_index'],md['userids']))
    dict2=dict(zip(md['shops_index'],md['shops']))
    nrecs['UserID']=nrecs['UserID_index'].map(dict1)
    nrecs['shops']=nrecs['Shop_index'].map(dict2)
    nrecs=nrecs.sort_values('UserID')
    nrecs.reset_index(drop=True, inplace=True)
    new=nrecs[['UserID','shops','Rating']]
    new['recommendations'] = list(zip(new.shops, new.Rating))
    res=new[['UserID','recommendations']]
    res_new=res['recommendations'].groupby([res.UserID]).apply(list).reset_index()

    #creating a new df for userid2552's collaborative filtering-derived recommendations
    collab_rec_2552 = pd.DataFrame(dict(res_new[res_new["UserID"]==2552]['recommendations'].tolist()[0]),index=[0]).T.sort_values(0,ascending=False)

    #creating a list of outlets userid2552 has rated earlier on
    rated_2552 = mbcf_try_pd[mbcf_try_pd['userids']==2552]['shops'].tolist()

    #filtering out those 10 outlets userid2552 has rated initially from the collaborative filtering recommendation list...
    collab_rankedrecs_2552 = collab_rec_2552.loc[[shop for shop in collab_rec_2552.index if shop not in rated_2552],0]

    #organizing the above series column into a df of recommendations and collaborative filtering rating predictions
    collab_2552_df = pd.DataFrame({'recommendations':collab_rankedrecs_2552.index,'collab_filter_predicted_ratings':collab_rankedrecs_2552})

    #reading in the previously prepped df meant for content-based filtering here for content-based filtering recommendations..
    content_f = pd.read_csv('content_based_df_nouser.csv')

    #merging userid2552's info with the df meant for content-based filtering so that rcontent-based filtering can make recommendations via rating predictions for userid 2552 later on...
    content_2552 = pd.merge(content_f,user_item_2552,how='left',on='shops')

    #getting dummies for categorical columns...
    content_2552_wdummies = pd.get_dummies(content_2552, columns=['shops','category_alias'], drop_first=False)

    #setting feature and target
    X = content_2552_wdummies.drop(['ratings'], axis=1)
    y = content_2552_wdummies['ratings']

    #collating dummified columns
    shops_cats_list = [col for col in content_2552_wdummies.columns if (col.startswith('shops')) or (col.startswith('category'))]

    #extending with review_count and rating
    shops_cats_list.extend(['review_count','rating','userids'])

    #as tfidf can only work on one column of texts at a time, am separating features as below...
    X1 = X['reviews']
    X2 = X[shops_cats_list]

    #Assigning a new variable name to X1 for processing.
    rev = X1

    #creating customized stop words' list
    cust_stop_words = [word for word in stop_words.ENGLISH_STOP_WORDS]

    #adding on to the above list based on preliminary word cloud EDA
    cust_stop_words.extend(["wa","ha","just","ve","did","got","quite"])

    #preprocessing text in reviews by defining a function to do so
    lemm = WordNetLemmatizer()

    def text_processer(raw_text):
        # Function to convert a raw string of text to a string of words
        # The input is a single string (a raw unprocessed text), and
        # the output is a single string (a preprocessed text)

        # 1. Remove http urls.
        review_text = re.sub("\(http.+\)", " ", raw_text)

        # 2. Remove non-letters.
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)

        # 3. Convert to lower case, split into individual words.
        words = letters_only.lower().split()

        # 4. Lemmatize words.
        lemmed_words = [lemm.lemmatize(i) for i in words]

        # 5. Remove stop words.

        meaningful_words = [w for w in lemmed_words if not w in cust_stop_words]

        # 6. Join the words back into one string separated by space,
        # and return the result.
        return(" ".join(meaningful_words))

    #showing how the processed reviews look like
    rev_processed = pd.Series([text_processer(text) for text in rev])

    #using tfidf vectorizer to convert the reviews into term frequency columns...
    tvec_naive = TfidfVectorizer(stop_words = cust_stop_words)  #instantiating TfidfVectorizer with customized stop words

    X1_tvec_naive = tvec_naive.fit_transform(rev_processed).todense()   #fitting tvec and transforming the processed reviews
    X1_tvec_naive_df = pd.DataFrame(X1_tvec_naive, columns = tvec_naive.get_feature_names())  #converting it into a dataframe for easy lookup.

    #combining tvec-df with the rest of the features for rating prediction for userid 2552 later on...
    X_legit = pd.concat([X1_tvec_naive_df,X2], axis=1)

    #adding back the column of ratings so that it can be dropped below-sorry sometimes my train of thought may sound illogical
    X_legit['ratings'] = y

    #creating X_train manually for userid 2552
    X_train_2552 = X_legit[X_legit['userids']==2552].drop(['ratings','userids'],axis=1)

    #creating y_train manually for userid 2552
    y_train_2552 = X_legit[X_legit['userids']==2552]['ratings']

    #creating X_test manually for userid 2552 which contains all outlets that have not been rated by userid 2552
    X_test_2552 = X_legit[X_legit['userids']!=2552].drop(['ratings','userids'],axis=1)

    #instantiate scaler since not all of the features are of the same scale, eg. review_count and rating
    ss= StandardScaler()

    #fitting the train and transforming both the train and test sets
    X_train_2552_sc = ss.fit_transform(X_train_2552)
    X_test_2552_sc = ss.transform(X_test_2552)

    #learning rate, max depth, and n_estimators were retrieved from a tuned xgb model (notebook on future plan for xgb) saved in the folder but in order to use random_state which was not used during tuning, I am just instantiating a new xgb instance with the 3 tuned hyperparams set accordingly...
    xgb = XGBClassifier(learning_rate=0.5, max_depth=9, n_estimators=200, random_state=42)

    #training the loaded model on the dataset containing the new user, userid 2552's ratings.
    xgb.fit(X_train_2552_sc, y_train_2552)

    #stacking X_test_2552 as first step in regenerating the shops column for predictions
    trial = X_test_2552.stack()

    #creating loop to re-generate original X_test_2552 order of shops
    index_lst = []
    outlets_lst = []
    for n in range(len(trial.index)):
        if trial.index[n][1].startswith('shops_') and trial[n]!=0:
            index_lst.append(str(trial.index[n][0]))
            outlets_lst.append(trial.index[n][1])
    index_lst = [int(x) for x in index_lst]
    reconstructed_X_test_2552 = pd.DataFrame({'shops':outlets_lst}, index=index_lst)

    #generating content-based filtering rating predictions for userid 2552
    rating_predictions = xgb.predict(X_test_2552_sc)

    #adding new column of rating predictions into the reconstructed X_test_2552
    reconstructed_X_test_2552['predicted_ratings']=rating_predictions

    #giving the reconstructed df a more easily understood name for distinction from the collaborative filtering df dealt with above
    content_2552_df = reconstructed_X_test_2552

    #trimming off the shops' prefixes so that they can eventually be merged with the collaborative filtering df
    content_2552_df['shops'] = content_2552_df['shops'].apply(lambda x: x[6:])

    #renaming the column of rating predictions to distinguish from collaborative filtering's prediction column later on when both dfs are merged.
    content_2552_df.rename(columns={'predicted_ratings':'content_filter_predicted_ratings'},inplace=True)

    #renaming collaborative filtering df's recommendations' column so that it can be merged with the content-based filtering df.
    collab_2552_df.rename(columns={'recommendations':'shops'},inplace=True)

    #reseting the index in the collaborative filtering df so that the index is numerical again
    collab_2552_df.reset_index(drop=True,inplace=True)

    #merging both content-based filtering and collaborating filtering df to prepare to make hybrid recommendations for userid 2552
    content_collab_2552_df = pd.merge(content_2552_df,collab_2552_df,how='inner',on='shops')

    #as mentioned in the previous sub-notebook on this hybrid recommender's evaluation, the following are the content-based and collaborative filtering's ratings' weights
    con_wt = 0.97 / (0.97 + 1.0)
    collab_wt = 1.0 / (0.97 + 1.0)

    #feature engineering to add hybrid recommender's rating predictions into the combined df by multiplying the respective rating predictions by weights based on both models' f1 scores derived from prior evaluation and summing them up to yield hybrid predictions
    content_collab_2552_df['final_weighted_rating_predictions'] = (content_collab_2552_df['content_filter_predicted_ratings']*con_wt) + (content_collab_2552_df['collab_filter_predicted_ratings']*collab_wt)

    #top 5 coffee-drinking outlet recommendations for userid 2552 (me!) based on my ratings given rather randomly to 10 of the outlets earlier on...
    #recommendations_top_5 = content_collab_2552_df.sort_values('final_weighted_rating_predictions',ascending=False).head()
    top_5_recs = content_collab_2552_df[['shops','final_weighted_rating_predictions']].sort_values('final_weighted_rating_predictions',ascending=False).head()
    top_5_recs.reset_index(drop=True,inplace=True)
    first = top_5_recs.loc[0,'shops']
    second = top_5_recs.loc[1,'shops']
    third = top_5_recs.loc[2,'shops']
    fourth = top_5_recs.loc[3,'shops']
    fifth = top_5_recs.loc[4,'shops']

    return render_template('outcome.html', first=first, second=second, third=third, fourth=fourth, fifth=fifth, shop1=shop1, rate1=rate1, shop2=shop2, rate2=rate2, shop3=shop3, rate3=rate3, shop4=shop4, rate4=rate4, shop5=shop5, rate5=rate5, shop6=shop6, rate6=rate6, shop7=shop7, rate7=rate7, shop8=shop8, rate8=rate8, shop9=shop9, rate9=rate9, shop10=shop10, rate10=rate10, url_alias=url_alias)





#@app.route('/add', methods=['POST'])
#def add_entry():
#    if not session.get('logged_in'):
#        abort(401)
#    db = get_db()
#    db.execute('insert into entries (title, text) values (?, ?)',
#                 [request.form['title'], request.form['text']])
#    db.commit()
#    flash('New entry was successfully posted')
#    return redirect(url_for('show_entries'))

#@app.route('/login', methods=['GET', 'POST'])
#def login():
#    error = None
#    if request.method == 'POST':
#        if request.form['username'] != app.config['USERNAME']:
#            error = 'Invalid username'
#        elif request.form['password'] != app.config['PASSWORD']:
#            error = 'Invalid password'
#        else:
#            session['logged_in'] = True
#            flash('You were logged in')
#            return redirect(url_for('show_entries'))
#    return render_template('login.html', error=error)

#@app.route('/logout')
# def logout():
#    session.pop('logged_in', None)
#    flash('You were logged out')
#    return redirect(url_for('show_entries'))
