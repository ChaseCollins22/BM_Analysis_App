from flask import Flask, render_template, request
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

def to_minutes(time):
        return (int(time[0:2]) * 60) + (int(time[3:5])) + (int(time[6:]) / 60)

def pace_to_minutes(time):
    return (int(time[0:2])) + (int(time[3:]) / 60)

def convert_age_division(division):
    return int(division[0:2] + division[3:])

def clean_runners_data(df):
    df['20K'].mask(df['20K'] == '–',inplace=True)
    df['15K'].mask(df['15K'] == '–',inplace=True)
    df['10K'].mask(df['10K'] == '–',inplace=True)
    df['5K'].mask(df['5K'] == '–',inplace=True)
    df = df.dropna(subset=['20K', '15K', '10K', '5K'])
    df = df.reset_index(drop=True)

    for column_name in df.columns[5:-1]:
        df[column_name] = df[column_name].apply(lambda time: to_minutes(time))
    df['Age_Division'] = df['Age_Division'].apply(lambda division: convert_age_division(division))
    return df

@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/data', methods=['GET', 'POST'])
def data():
    df_countries = pd.read_csv('data/2023_BM_Country_Data.csv', index_col=0)
    df_women = pd.read_csv('data/2023_BM_Female_Data.csv', index_col=0)
    df_men = pd.read_csv('data/2023_BM_Male_Data.csv', index_col=0)

    countries = df_countries['Country_Name'].unique().tolist()
    athletes = df_countries['Num_Athletes'].values.tolist()
    
    age_divisions = pd.concat([df_men['Age_Division'], df_women['Age_Division']], axis=1, keys=['Men', 'Women'])
    mens_ages = age_divisions['Men'].value_counts().to_list()
    womens_ages = age_divisions['Women'].value_counts().to_list()
    index = df_women['Age_Division'].unique()
    ages = pd.DataFrame({'Men':mens_ages, 'Women':womens_ages}, index=index)

    columns = ages.index.to_list()
    men_age_div_count = ages['Men'].values.tolist()
    women_age_div_count = ages['Women'].values.tolist()

    total_men = int(df_men['Gender'].count())
    total_women = int(df_women['Gender'].count())


    # Drop unecessary columns from mens csv
    df_men = df_men.drop(columns=['25.2 Miles', '24 Miles', '23 Miles', '20 Miles', '21 Miles', 'Name', 'Net_Finish_Time',
                            'State', 'Bib_Number', 'Overall_Pace'])

    # Add 'Female' column with 0 = False and change gender to 'Male; with 1 = True
    df_men = df_men.rename(columns={'Gender': 'Male'})
    df_men['Male'] = 1
    df_men['Female'] = 0

    # # Drop specific outlier
    # df_men.drop(14062, inplace=True)

    # Clean and convert splits from HH:MM:SS to decimal 
    df_men = clean_runners_data(df_men)

    # Drop unecessary columns from womens csv
    df_women = df_women.drop(columns=['25.2 Miles', '24 Miles', '23 Miles', '20 Miles', '21 Miles', 'Name', 'Net_Finish_Time',
                         'State', 'Bib_Number', 'Overall_Pace'])

    # Add 'Female' column with 0 = False and change gender to 'Male; with 1 = True
    df_women = df_women.rename(columns={'Gender': 'Female'})
    df_women['Male'] = 0
    df_women['Female'] = 1

    # Clean and convert splits from HH:MM:SS to decimal 
    df_women = clean_runners_data(df_women)
    # Merge men and womens dataframes 
    df_all = pd.concat([df_men, df_women]).reset_index(drop=True)

    df_all.drop(14061, inplace=True)
    df_all.drop(11548, inplace=True)
    df_all.drop(14938, inplace=True)
    df_all.drop(26322, inplace=True)
    df_all.drop(df_all.loc[(df_all['5K'] < 25) & (df_all['Full'] > 300)].index, inplace=True)
    df_all.drop(df_all.loc[(df_all['5K'] < 18) & (df_all['Full'] > 240)].index, inplace=True)
    df_all.drop(df_all[(df_all['5K'] > 36) & (df_all['Full'] < 275)].index, inplace=True)
    df_all.drop(df_all[(df_all['5K'] > 41) & (df_all['Full'] < 350)].index, inplace=True)
    df_all.drop(df_all[df_all['5K'] > 42].index, inplace=True)
    df_all.drop(df_all[df_all['Full'] > 380].index, inplace=True)
    df_all.drop(df_all[(df_all['Full'] > 260) & (df_all['5K'] < 24)].index, inplace=True)
    df_all.drop(df_all[(df_all['Full'] > 310) & (df_all['5K'] < 30)].index, inplace=True)
    df_all.drop(df_all[(df_all['Full'] > 280) & (df_all['5K'] < 26)].index, inplace=True)
    df_all['5K'] = df_all['5K'] - 1.5

    X = df_all.drop(columns=['Place_Division', 'Full','Place_Gender','Place_Overall','10K','15K','20K','Half','25K','30K','35K','40K'])
    y = df_all[['Full']].copy()
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42, shuffle=True)

    rf_regr = RandomForestRegressor(max_depth=25, random_state=42)

    rf_regr.fit(X_train, y_train.values.ravel())
    rf_pred_y = rf_regr.predict(X_test)

    five_K_values = X_test[['5K']]['5K'].to_list()
    full_values = y_test['Full'].to_list()
    predicted_y_values = rf_pred_y.tolist()
    pred_values = list(zip(five_K_values, predicted_y_values))
    real_values = list(zip(five_K_values, full_values))



    linear_regr = LinearRegression()
    # Train the model using the training sets
    linear_regr.fit(X_train, y_train)
    linear_regr_pred = linear_regr.predict(X_test)
    linear_regr_pred_y = linear_regr_pred.tolist()

    lr_pred_values = list(zip(five_K_values, linear_regr_pred_y))
    lr_real_values = list(zip(five_K_values,full_values))

    if request.method == "POST":
        #Get form Data
        five_k_mins = request.form.get("fivekMins")
        five_k_seconds = request.form.get("fivekSeconds")
        age = request.form.get("age")
        gender = request.form.get("gender")

        #Clean and Convert form data
        five_k_seconds = int(five_k_seconds) / 60
        five_k_time = round(int(five_k_mins) + five_k_seconds, 2)

        if gender == "M":
            male = 1
            female = 0
            gender = "Male"
        else:
            female = 1
            male = 0
            gender = "Female"
        age = int(age)
        if age >= 18 and age <= 39:
            age_div = 1839
        elif age > 39 and age <= 44:
            age_div = 4044
        elif age > 44 and age <= 49:
            age_div = 4549
        elif age > 49 and age <= 54:
            age_div = 5054
        elif age > 54 and age <= 59:
            age_div = 5559
        elif age > 59 and age <= 64:
            age_div = 6064
        elif age > 64 and age <= 69:
            age_div = 6569
        elif age > 69 and age <= 74:
            age_div = 7074
        elif age > 74 and age <= 79:
            age_div = 7579
        else:
            age_div = 8099

        data = [age_div, male, five_k_time, female]

        predict_df = pd.DataFrame([data], columns = ['Age_Division', 'Male', '5K', 'Female'])
        predicted_time = rf_regr.predict(predict_df)
        predicted_time = str(predicted_time)[1:-1].split('.')
        predicted_hours = int((int(predicted_time[0]) / 60))
        predicted_minutes = ((int(predicted_time[0]) % 60))
        predicted_seconds = int((int(predicted_time[1][:2]) / 100) * 60)
        final_predicted_time = str(predicted_hours) + ":" + str(predicted_minutes) + ":" + str(predicted_seconds)
    
        return render_template("prediction.html", predicted_time=json.dumps(final_predicted_time), five_k_mins=json.dumps(five_k_mins),
                                five_k_seconds=json.dumps(five_k_seconds), age=json.dumps(age), gender=json.dumps(gender))
 
    return render_template("data.html",athletes=json.dumps(athletes), countriesData=json.dumps(countries), columns=json.dumps(columns),
                            men_data=json.dumps(men_age_div_count), women_data=json.dumps(women_age_div_count), total_men=json.dumps(total_men),
                            total_women=json.dumps(total_women), five_k_values=json.dumps(five_K_values), full_values=json.dumps(real_values),
                            pred_values=json.dumps(pred_values), lr_pred_values=json.dumps(lr_pred_values), lr_real_values=json.dumps(lr_real_values))
@app.route('/data/predict')
def predict():
    return render_template('prediction.html')

if __name__ == "__main__":
    app.run(debug=True)
    
