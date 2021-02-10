# import pkgs
import streamlit as st 
import datetime
from collections import UserString

# EDA pkgs
from numpy.lib.function_base import vectorize
from pandas.core.algorithms import value_counts
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('Agg')
import os
import joblib
from PIL import Image

# DB
import sqlite3
from db_functions import *

PAGE_CONFIG = {'page_title':'Salary Estimator', 'page_icon':':money_with_wings:', 'layout':'centered'}
st.set_page_config(**PAGE_CONFIG)
st.set_option('deprecation.showPyplotGlobalUse', False)

# load a model
def load_ml(ml_file):
    model = joblib.load(open(os.path.join(ml_file), 'rb'))
    return model

def get_value(val,my_dict):
	for key ,value in my_dict.items():
		if val == key:
			return value

# Find the Key From Dictionary
def get_key(val,my_dict):
	for key ,value in my_dict.items():
		if val == value:
			return key
class Monitor(object):
    """docstring for Monitor"""
    
    """ 
    create a database connection to the SQLite database specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    db_file = "data/data.db"
    conn = None

    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        pass


    def __init__(self,age=None ,workclass=None ,fnlwgt=None ,education=None ,education_num=None ,marital_status=None ,occupation=None ,relationship=None ,race=None ,sex=None ,capital_gain=None ,capital_loss=None ,hours_per_week=None ,native_country=None,predicted_class=None,model_class=None,time_of_prediction=None):
        super(Monitor, self).__init__()
        self.age = age
        self.workclass = workclass
        self.fnlwgt = fnlwgt
        self.education = education
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.native_country = native_country
        self.predicted_class = predicted_class
        self.model_class = model_class
        self.time_of_prediction=time_of_prediction

    def __repr__(self):
        "Monitor(age = {self.age},workclass = {self.workclass},fnlwgt = {self.fnlwgt},education = {self.education},education_num = {self.education_num},marital_status = {self.marital_status},occupation = {self.occupation},relationship = {self.relationship},race = {self.race},sex = {self.sex},capital_gain = {self.capital_gain},capital_loss = {self.capital_loss},hours_per_week = {self.hours_per_week},native_country = {self.native_country},predicted_class = {self.predicted_class},model_class = {self.model_class}, time_of_prediction={self.time_of_prediction})".format(self=self)

    def create_prediction_table(self):
        """ create a prediction table """
        try:
            self.c.execute("""CREATE TABLE IF NOT EXISTS predtable
            (age NUMERIC,workclass TEXT,fnlwgt NUMERIC,education TEXT,education_num NUMERIC,marital_status TEXT,occupation TEXT,relationship TEXT,race TEXT,sex TEXT,capital_gain NUMERIC,capital_loss NUMERIC,hours_per_week NUMERIC,native_country TEXT,predicted_class NUMERIC,model_class TEXT, time_of_prediction TEXT)""")

        except Exception as e:
            pass

    def add_pred_data(self):
        self.c.execute("""INSERT INTO predtable
        (age ,workclass ,fnlwgt ,education ,education_num ,marital_status ,occupation ,relationship ,race ,sex ,capital_gain ,capital_loss ,hours_per_week ,native_country,predicted_class,model_class, time_of_prediction) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.native_country,self.predicted_class,self.model_class, self.time_of_prediction))        
        self.conn.commit()

    def view_all_pred_data(self):
        self.c.execute('SELECT * FROM predtable')
        data = self.c.fetchall()
        return data

# Layout templates
html_temp = """
<div style="background-color:{};padding:10px;border-radius:10px">
<h1 style="color:{};text-align:center;">Salary Estimator</h1>
</div>
"""


def main():
    """Salary Estimator"""
    st.markdown(html_temp.format('#ff9090','white'), unsafe_allow_html=True)

    # load file
    df = pd.read_csv('data/adult_salary.csv')

    menu = ['EDA', 'Prediction', 'Countries', 'Metrics']
    choice = st.sidebar.selectbox("Menu", menu)

    # Dictionary of Mapped Values
    d_workclass = {"Never-worked": 0, "Private": 1, "Federal-gov": 2, "?": 3, "Self-emp-inc": 4, "State-gov": 5, "Local-gov": 6, "Without-pay": 7, "Self-emp-not-inc": 8}
    d_education = {"Some-college": 0, "10th": 1, "Doctorate": 2, "1st-4th": 3, "12th": 4, "Masters": 5, "5th-6th": 6, "9th": 7, "Preschool": 8, "HS-grad": 9, "Assoc-acdm": 10, "Bachelors": 11, "Prof-school": 12, "Assoc-voc": 13, "11th": 14, "7th-8th": 15}
    d_marital_status = {"Separated": 0, "Married-spouse-absent": 1, "Married-AF-spouse": 2, "Married-civ-spouse": 3, "Never-married": 4, "Widowed": 5, "Divorced": 6}
    d_occupation = {"Tech-support": 0, "Farming-fishing": 1, "Prof-specialty": 2, "Sales": 3, "?": 4, "Transport-moving": 5, "Armed-Forces": 6, "Other-service": 7, "Handlers-cleaners": 8, "Exec-managerial": 9, "Adm-clerical": 10, "Craft-repair": 11, "Machine-op-inspct": 12, "Protective-serv": 13, "Priv-house-serv": 14}
    d_relationship = {"Other-relative": 0, "Not-in-family": 1, "Own-child": 2, "Wife": 3, "Husband": 4, "Unmarried": 5}
    d_race = {"Amer-Indian-Eskimo": 0, "Black": 1, "White": 2, "Asian-Pac-Islander": 3, "Other": 4}
    d_sex = {"Female": 0, "Male": 1}
    d_native_country = {"Canada": 0, "Philippines": 1, "Thailand": 2, "Scotland": 3, "Germany": 4, "Portugal": 5, "India": 6, "China": 7, "Japan": 8, "Peru": 9, "France": 10, "Greece": 11, "Taiwan": 12, "Laos": 13, "Hong": 14, "El-Salvador": 15, "Outlying-US(Guam-USVI-etc)": 16, "Yugoslavia": 17, "Cambodia": 18, "Italy": 19, "Honduras": 20, "Puerto-Rico": 21, "Dominican-Republic": 22, "Vietnam": 23, "Poland": 24, "Hungary": 25, "Holand-Netherlands": 26, "Ecuador": 27, "South": 28, "Guatemala": 29, "United-States": 30, "Nicaragua": 31, "Trinadad&Tobago": 32, "Cuba": 33, "Jamaica": 34, "Iran": 35, "?": 36, "Haiti": 37, "Columbia": 38, "Mexico": 39, "England": 40, "Ireland": 41}
    d_class = {">50K": 0, "<=50K": 1}
    
    # EDA
    if choice == 'EDA':
        st.subheader('Exploratory Data Analysis (EDA)')
        task = ['Choose a task', 'Dataset Descripctions', 'Preview Dataset', 'Plot Correlation']
        task_option = st.selectbox("Task", task)
        
      
        # description
        if task_option == 'Dataset Descripctions':
            st.write(df.describe())
            col_1, col_2 = st.beta_columns(2)
            # shape
            if col_1.checkbox('Dataset Shape'):
                dimesion = col_1.radio('Dimension by',('Shape','Columns', 'Rows'))

                if dimesion == 'Shape':
                    col_1.text('The shape of dataset')
                    col_1.write(df.shape)

                elif dimesion == 'Columns':
                    col_1.text('The number of Columns')
                    col_1.write(df.shape[1])

                elif dimesion == 'Rows':
                    col_1.text('The number of Rows')
                    col_1.write(df.shape[0])      

            # show columns
            if col_2.checkbox('Column Names'):
                col_2.write(df.columns)

        # data
        elif task_option == 'Preview Dataset':
            num = st.number_input('How many data do you want to see:', value=0)
            st.dataframe(df.head(num))
            col_1, col_2 = st.beta_columns(2)
            selection = col_1.radio('Selection by',('Columns', 'Rows', 'Value counts'))
                    # selections
            if selection == 'Columns':
                columns = df.columns.tolist()
                selected_columns = col_2.multiselect("Select Columns", columns)
                c_df = df[selected_columns]
                st.dataframe(c_df)

            elif selection == 'Rows':
                selected_rows = col_2.multiselect("Select Rows", df.head(num).index)
                r_df = df.loc[selected_rows]
                st.dataframe(r_df)

            else:
                st.text("Values counts by class")
                st.write(df.iloc[:,-1].value_counts())

        elif task_option == 'Plot Correlation':

            if st.checkbox("Show correlation plot (Matplotlib)"):
                plt.matshow(df.corr())
                st.pyplot()

            if st.checkbox("Show correlation plot (Seaborn)"):
                st.write(sns.heatmap(df.corr()))
                st.pyplot()

    # Predict
    elif choice == 'Prediction':
        st.subheader('Prediction')    
                
        # RECEIVE USER INPUT
        age = st.slider("Select Age",16,90)
        workclass = st.selectbox("Select Work Class",tuple(d_workclass.keys()))
        fnlwgt = st.number_input("Enter FNLWGT",1.228500e+04,1.484705e+06)
        education = st.selectbox("Select Education",tuple(d_education.keys()))
        education_num = st.slider("Select Education Level",1,16)
        marital_status = st.selectbox("Select Marital-status",tuple(d_marital_status.keys()))
        occupation = st.selectbox("Select Occupation",tuple(d_occupation.keys()))
        relationship = st.selectbox("Select Relationship",tuple(d_relationship.keys()))
        race = st.selectbox("Select Race",tuple(d_race.keys()))
        sex = st.radio("Select Sex",tuple(d_sex.keys()))
        capital_gain = st.number_input("Enter Capital Gain", value=0, min_value=0, max_value=99999)
        capital_loss = st.number_input("Enter Capital Loss", value=0, min_value=0, max_value=4356)
        hours_per_week = st.number_input("Enter Hrs Per Week ", value=0, min_value=0, max_value=99)
        native_country = st.selectbox("Select Native Country",tuple(d_native_country.keys()))
        # USER INPUT ENDS HERE

        # GET VALUES FOR EACH INPUT
        k_workclass = get_value(workclass,d_workclass)
        k_education = get_value(education,d_education)
        k_marital_status = get_value(marital_status,d_marital_status)
        k_occupation = get_value(occupation,d_occupation)
        k_relationship = get_value(relationship,d_relationship)
        k_race = get_value(race,d_race)
        k_sex = get_value(sex,d_sex)
        k_native_country = get_value(native_country,d_native_country)

        # RESULT OF USER INPUT
        options = [age, workclass, fnlwgt, education ,education_num ,marital_status ,occupation ,relationship ,race ,sex ,capital_gain ,capital_loss ,hours_per_week ,native_country]
        vectorized_result = [age, k_workclass, fnlwgt, k_education, education_num, k_marital_status, k_occupation, k_relationship ,k_race ,k_sex ,capital_gain ,capital_loss ,hours_per_week ,k_native_country]
        data_reshape = np.array(vectorized_result).reshape(1, -1)
        st.info(options)
        st.text("Using this encoding for prediction")
        st.success(vectorized_result)

        prettified_result = {"age":age,
        "workclass":workclass,
        "fnlwgt":fnlwgt,
        "education":education,
        "education_num":education_num,
        "marital_status":marital_status,
        "occupation":occupation,
        "relationship":relationship,
        "race":race,
        "sex":sex,
        "capital_gain":capital_gain,
        "capital_loss":capital_loss,
        "hours_per_week":hours_per_week,
        "native_country":native_country}

        st.subheader("Prettify JSON")
        st.json(prettified_result)

        # prediction
        st.subheader("Prediction")
        if st.checkbox("Predict"):
            ml_list = ['Logistic Regression', 'Random Forest', 'Naive Bayes']
        
            # select model
            choice_ml = st.selectbox("Select ML model", ml_list)
            if st.button("Predict"):

                if choice_ml == 'Logistic Regression':
                    model = load_ml('models/salary_logit_model.pkl')

                elif choice_ml == 'Random Forest':
                    model = load_ml('models/salary_rf_model.pkl')
                
                elif choice_ml == 'Naive Bayes':
                    model = load_ml('models/salary_nv_model.pkl')
                
                pred = model.predict(data_reshape)
                pred_result = get_key(pred, d_class)
                time_of_prediction = datetime.datetime.now()
                monitor = Monitor(age ,workclass ,fnlwgt ,education ,education_num ,marital_status ,occupation ,relationship ,race ,sex ,capital_gain ,capital_loss ,hours_per_week ,native_country,pred_result, model, time_of_prediction)
                monitor.create_prediction_table()
                monitor.add_pred_data(age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,pred_result,model, time_of_prediction)
                st.success('Predicted Salary as {}'.format(pred_result))
    
    # Countries
    elif choice == 'Countries':
        st.subheader('Countries')     

        # lsit of countries
        choice_country = st.selectbox('Choose a country', tuple(d_native_country.keys()))
        # disply selected countries 
        st.text(choice_country)  
        df1 = pd.read_csv('data/adult_salary_data.csv')
        choice_country_df = df1[df1['native-country'].str.contains(choice_country)]
        st.dataframe(choice_country_df.head(10))

        countries_img = {'af': 'Afghanistan','al': 'Albania','dz': 'Algeria','as': 'American Samoa','ad': 'Andorra','ao': 'Angola','ai': 'Anguilla','aq': 'Antarctica','ag': 'Antigua And Barbuda','ar': 'Argentina','am': 'Armenia','aw': 'Aruba','au': 'Australia','at': 'Austria','az': 'Azerbaijan','bs': 'Bahamas','bh': 'Bahrain','bd': 'Bangladesh','bb': 'Barbados','by': 'Belarus','be': 'Belgium','bz': 'Belize','bj': 'Benin','bm': 'Bermuda','bt': 'Bhutan','bo': 'Olivia','ba': 'Bosnia And Herzegovina','bw': 'Botswana','bv': 'Bouvet Island','br': 'Brazil','io': 'British Indian Ocean Territory','bn': 'Brunei Darussalam','bg': 'Bulgaria','bf': 'Burkina Faso','bi': 'Burundi','kh': 'Cambodia','cm': 'Cameroon','ca': 'Canada','cv': 'Cape Verde','ky': 'Cayman Islands','cf': 'Central African Republic','td': 'Chad','cl': 'Chile','cn': "People'S Republic Of China",'cx': 'Hristmas Island','cc': 'Cocos (Keeling) Islands','co': 'Colombia','km': 'Comoros','cg': 'Congo','cd': 'Congo, The Democratic Republic Of','ck': 'Cook Islands','cr': 'Costa Rica','ci': "Côte D'Ivoire",'hr': 'Croatia','cu': 'Cuba','cy': 'Cyprus','cz': 'Czech Republic','dk': 'Denmark','dj': 'Djibouti','dm': 'Dominica','do': 'Dominican Republic','ec': 'Ecuador','eg': 'Egypt','eh': 'Western Sahara','sv': 'El Salvador','gq': 'Equatorial Guinea','er': 'Eritrea','ee': 'Estonia','et': 'Ethiopia','fk': 'Falkland Islands (Malvinas)','fo': 'Aroe Islands','fj': 'Fiji','fi': 'Finland','fr': 'France','gf': 'French Guiana','pf': 'French Polynesia','tf': 'French Southern Territories','ga': 'Gabon','gm': 'Gambia','ge': 'Georgia','de': 'Germany','gh': 'Ghana','gi': 'Gibraltar','gr': 'Greece','gl': 'Greenland','gd': 'Grenada','gp': 'Guadeloupe','gu': 'Guam','gt': 'Guatemala','gn': 'Guinea','gw': 'Guinea-Bissau','gy': 'Guyana','ht': 'Haiti','hm': 'Heard Island And Mcdonald Islands','hn': 'Honduras','hk': 'Hong Kong','hu': 'Hungary','is': 'Iceland','in': 'India','id': 'Indonesia','ir': 'Iran, Islamic Republic Of','iq': 'Iraq','ie': 'Ireland','il': 'Israel','it': 'Italy','jm': 'Jamaica','jp': 'Japan','jo': 'Jordan','kz': 'Kazakhstan','ke': 'Kenya','ki': 'Kiribati','kp': "Korea, Democratic People'S Republic Of",'kr': 'Korea, Republic Of','kw': 'Kuwait','kg': 'Kyrgyzstan','la': "Lao People'S Democratic Republic",'lv': 'Latvia','lb': 'Lebanon','ls': 'Lesotho','lr': 'Liberia','ly': 'Libyan Arab Jamahiriya','li': 'Liechtenstein','lt': 'Lithuania','lu': 'Luxembourg','mo': 'Macao','mk': 'Macedonia, The Former Yugoslav Republic Of','mg': 'Madagascar','mw': 'Malawi','my': 'Malaysia','mv': 'Maldives','ml': 'Mali','mt': 'Malta','mh': 'Marshall Islands','mq': 'Martinique','mr': 'Mauritania','mu': 'Mauritius','yt': 'Mayotte','mx': 'Mexico','fm': 'Micronesia, Federated States Of','md': 'Moldova, Republic Of','mc': 'Monaco','mn': 'Mongolia','ms': 'Montserrat','ma': 'Morocco','mz': 'Mozambique','mm': 'Myanmar','na': 'Namibia','nr': 'Nauru','np': 'Nepal','nl': 'Netherlands','an': 'Netherlands Antilles','nc': 'New Caledonia','nz': 'New Zealand','ni': 'Nicaragua','ne': 'Niger','ng': 'Nigeria','nu': 'Niue','nf': 'Norfolk Island','mp': 'Northern Mariana Islands','no': 'Norway','om': 'Oman','pk': 'Pakistan','pw': 'Palau','ps': 'Palestinian Territory, Occupied','pa': 'Panama','pg': 'Papua New Guinea','py': 'Paraguay','pe': 'Peru','ph': 'Philippines','pn': 'Pitcairn','pl': 'Poland','pt': 'Portugal','pr': 'Puerto Rico','qa': 'Qatar','re': 'Réunion','ro': 'Romania','ru': 'Russian Federation','rw': 'Rwanda','sh': 'Saint Helena','kn': 'Saint Kitts And Nevis','lc': 'Saint Lucia','pm': 'Saint Pierre And Miquelon','vc': 'Saint Vincent And The Grenadines','ws': 'Samoa','sm': 'San Marino','st': 'Sao Tome And Principe','sa': 'Saudi Arabia','sn': 'Senegal','cs': 'Serbia And Montenegro','sc': 'Seychelles','sl': 'Sierra Leone','sg': 'Singapore','sk': 'Slovakia','si': 'Slovenia','sb': 'Solomon Islands','so': 'Somalia','za': 'South Africa','gs': 'South Georgia And South Sandwich Islands','es': 'Spain','lk': 'Sri Lanka','sd': 'Sudan','sr': 'Suriname','sj': 'Svalbard And Jan Mayen','sz': 'Swaziland','se': 'Sweden','ch': 'Switzerland','sy': 'Syrian Arab Republic','tw': 'Taiwan, Republic Of China','tj': 'Tajikistan','tz': 'Tanzania, United Republic Of','th': 'Thailand','tl': 'Timor-Leste','tg': 'Togo','tk': 'Tokelau','to': 'Tonga','tt': 'Trinidad And Tobago','tn': 'Tunisia','tr': 'Turkey','tm': 'Turkmenistan','tc': 'Turks And Caicos Islands','tv': 'Tuvalu','ug': 'Uganda','ua': 'Ukraine','ae': 'United Arab Emirates','gb': 'United Kingdom','us': 'United States','um': 'United States Minor Outlying Islands','uy': 'Uruguay','uz': 'Uzbekistan','ve': 'Venezuela','vu': 'Vanuatu','vn': 'Viet Nam','vg': 'British Virgin Islands','vi': 'U.S. Virgin Islands','wf': 'Wallis And Futuna','ye': 'Yemen','zw': 'Zimbabwe'}

        for k, v in countries_img.items():
            if v == choice_country:
                temp_img = 'imgs/flags/{}.png'.format(k)
                img_flag = Image.open(os.path.join(temp_img)).convert('RGB')
                st.image(img_flag, caption='{}.format(v)', use_column_width=True)

        if st.checkbox('Choose columns to show'):
            columns_df = choice_country_df.columns.tolist()
            columns_countries = st.multiselect('Choose column', columns_df)
            df2 = df[columns_countries]
            st.dataframe(df2.head(10))

            if st.button('Plot country'):
                st.area_chart(df2)
                st.pyplot()

    # Metrics
    elif choice == 'Metrics':
        st.subheader('Metrics')
        db = sqlite3.connect('data.db')
        df3 = pd.read_sql_query('SELECT * FROM predtable', db)    
        st.dataframe(df3)

       
if __name__ == "__main__":
    main()

