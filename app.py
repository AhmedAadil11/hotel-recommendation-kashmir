from flask import Flask, render_template, request
import numpy as np
import pickle as pkl

app = Flask(__name__)

# Load the trained model and encoder
with open('decision_tree.pkl', 'rb') as f:
    model = pkl.load(f)
with open('ohe.pkl', 'rb') as f:
    ohe = pkl.load(f)

"""
 0   user_budget_total_inr          18000 non-null  int64  
 1   user_daily_budget_inr          18000 non-null  int64  
 2   stay_days                      18000 non-null  int64  
 3   family_count                   18000 non-null  int64  
 4   adults_count                   18000 non-null  int64  
 5   children_count                 18000 non-null  int64  
 6   rooms_required                 18000 non-null  int64  
 7   travel_season                  18000 non-null  object 
 8   trip_purpose                   18000 non-null  object 
 9   district                       18000 non-null  object 
 10  star_category                  18000 non-null  int64  
 11  has_mountain_or_lake_view      18000 non-null  int64  
 12  distance_to_sxr_airport_km     18000 non-null  float64
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    error_message = ""

    if request.method == 'POST':
        try:
            # 1. Numerical / binary features
            user_budget_total_inr = float(request.form['user_budget_total_inr'])
            user_daily_budget_inr = float(request.form['user_daily_budget_inr'])
            stay_days = float(request.form['stay_days'])
            family_count = float(request.form['family_count'])
            adults_count = float(request.form['adults_count'])
            children_count = float(request.form['children_count'])
            rooms_required = float(request.form['rooms_required'])
            star_category = float(request.form['star_category'])
            has_mountain_or_lake_view = float(request.form['has_mountain_or_lake_view'])
            distance_to_sxr_airport_km = float(request.form['distance_to_sxr_airport_km'])

            # 2. Categorical features
            travel_season = request.form['travel_season']
            trip_purpose = request.form['trip_purpose']
            district = request.form['district']

            # 3. Numerical array only, with no text columns in it
            nums = [[
                user_budget_total_inr, user_daily_budget_inr, stay_days, family_count,
                adults_count, children_count, rooms_required,
                star_category, has_mountain_or_lake_view, distance_to_sxr_airport_km
            ]]

            # 4. Encode categorical features using ohe.pkl
            cat = ohe.transform([[travel_season, trip_purpose, district]])
            if hasattr(cat, 'toarray'):      # sparse -> dense
                cat = cat.toarray()

            # 5. Combine and predict
            input_data = np.hstack([nums, cat])
            prediction = model.predict(input_data)[0]

        except Exception as e:
            error_message = str(e)

    return render_template('index.html', prediction=prediction, error=error_message)

if __name__ == '__main__':
    app.run(debug=True)