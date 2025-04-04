from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# ============ Content based Filtering ============

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("main/aug_ind.csv", encoding='utf-8')

df["Weekly Off"] = df["Weekly Off"].fillna("No")

df['Features'] = df['Type'] + " " + df['Significance'] + " " + df['Best Time to visit']
df['Features'] += " " + df['Google review rating'].astype(str) + " " + df['Number of google review in lakhs'].astype(str) + " " + df['time needed to visit in hrs'].astype(str)

vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(df['Features'])

cosine_sim = cosine_similarity(feature_matrix)

def recommend_place_loc(place_name, df, cosine_sim, top_n=10):
    if place_name not in df['Name'].values:
        return pd.DataFrame()
    
    idx = df[df['Name'] == place_name].index[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    place_indices = [i[0] for i in sim_scores if i[0] != idx]
    
    selected_columns = ['State', 'City', 'Name', 'Type', 'Establishment Year', 
                        'Entrance Fee in INR', 'Weekly Off', 'Best Time to visit', 'Airport with 50km Radius']
    
    unique_places = df.iloc[place_indices][selected_columns]
    unique_places = unique_places[unique_places['Name'] != place_name]
    unique_places = unique_places.drop_duplicates(subset=['Name']).copy()
    
    return unique_places.head(top_n)



# ============ Feature based model ============

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("main/aug_ind.csv", encoding='utf-8')
df["Weekly Off"] = df["Weekly Off"].fillna("No")

features = ["State", "City", "Type", "time needed to visit in hrs", "Google review rating", "Best Time to visit"]
target = "Name"

df_encoded = df.copy()
label_encoders = {}

for col in ["State", "City", "Type", "Best Time to visit"]:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str).str.strip())
    label_encoders[col] = le

X = df_encoded[features]
y = df["Name"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

def recommend_place_feat(user_input):
    def safe_encode(value, column):
        if value in label_encoders[column].classes_:
            return label_encoders[column].transform([value])[0]
        else:
            raise ValueError(f"Error: '{value}' is not in training data for column '{column}'.")

    try:
        input_data = [
            safe_encode(user_input["State"], "State"),
            safe_encode(user_input["City"], "City"),
            safe_encode(user_input["Type"], "Type"),
            user_input["time needed to visit in hrs"],
            user_input["Google review rating"],
            safe_encode(user_input["Best Time to visit"], "Best Time to visit")
        ]

        input_df = pd.DataFrame([input_data], columns=features)

        predicted_names = rf_model.predict(X)
        recommended_places = df[df["Name"].isin(predicted_names)].copy()

        state = user_input["State"]
        city = user_input["City"]
        type_ = user_input["Type"]

        filters = [
            ((recommended_places["State"] == state) & (recommended_places["City"] == city) & (recommended_places["Type"] == type_)),
            ((recommended_places["State"] == state) & (recommended_places["City"] == city)),
            ((recommended_places["State"] == state) & (recommended_places["Type"] == type_)),
            ((recommended_places["Type"] == type_)),
            (recommended_places["Name"].notna())
        ]

        final_places = pd.DataFrame()

        for f in filters:
            new_places = recommended_places[f].drop_duplicates(subset=["Name"])
            final_places = pd.concat([final_places, new_places])
            final_places = final_places.drop_duplicates(subset=["Name"])
            if len(final_places) >= 10:
                break

        final_places = final_places.head(10)

        final_output = final_places[[
            "State", "City", "Name", "Type", "Establishment Year",
            "Entrance Fee in INR", "Weekly Off", "Best Time to visit", 
            "Airport with 50km Radius"
        ]]

        final_output["Establishment Year"] = final_output["Establishment Year"].astype(str).replace("-3500", "Unknown")
        final_output = final_output.reset_index(drop=True)

        return final_output

    except ValueError as e:
        return pd.DataFrame({"Error": [str(e)]})

# ============ Routes ============

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if request.form.get("location"):
            location = request.form["location"].strip().title()
            recommendations = recommend_place_loc(location, df, cosine_sim)

        else:
            try:
                user_input = {
                    "State": request.form.get("state", "").strip().title(),
                    "City": request.form.get("city", "").strip().title(),
                    "Type": request.form.get("type", "").strip().title(),
                    "time needed to visit in hrs": float(request.form.get("time_needed", 0)),
                    "Google review rating": float(request.form.get("rating", 0)),
                    "Best Time to visit": request.form.get("best_time", "").strip()
                }
                recommendations = recommend_place_feat(user_input)

            except ValueError as ve:
                return render_template("index.html", error=f"Input error: {ve}")

        if recommendations.empty:
            return render_template("index.html", error="No results found. Try different inputs.")
        else:
            return render_template(
                "results.html",
                tables=[recommendations.to_html(classes='data', index=False)],
                titles=recommendations.columns.values
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
