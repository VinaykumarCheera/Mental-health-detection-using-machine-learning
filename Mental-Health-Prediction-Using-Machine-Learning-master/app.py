from flask import Flask, url_for, redirect, render_template, request
from pandas import DataFrame
import pickle

# Load model and encoders
model = pickle.load(open("./model.pkl", 'rb'))
ct = pickle.load(open("./ct.pkl", "rb"))
le = pickle.load(open("./le.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.get('/form')
def show_form():
    return render_template('form.html')

@app.post('/submit_form')
def submit_form():
    try:
        print("Form Data Received:", request.form)  # Debugging: See all received form data

        name = request.form.get("inputName", "").strip()
        if not name:
            return "Error: Name is required.", 400

        # List of expected input fields
        form_fields = [
            "inputAge", "inputGender", "inputSelfEmployed", "inputFamilyHistory",
            "inputWorkInterference", "inputNoOfEmp", "inputRemoteWork", "inputTechCompany",
            "inputBenefits", "inputCareOptions", "inputWellnessProgram", "inputSeekHelp",
            "inputAnonymity", "inputLeave", "inputMentalHealthConsequence",
            "inputPhysHealthConsequence", "inputCoworkers", "inputSupervisor",
            "inputMentalHealthInterview", "inputPhysHealthInterview",
            "inputMentalVsPhysical", "inputObsConsequence"
        ]

        # Validate each field
        for field in form_fields:
            value = request.form.get(field, "").strip()
            if not value:
                message= f"Error: {field} is missing or empty."
                return render_template("error.html", name=name, message=message)
        # Prepare input data
        data = [{
            "Age": int(request.form["inputAge"]),
            "Gender": request.form["inputGender"],
            "self_employed": request.form["inputSelfEmployed"],
            "family_history": request.form["inputFamilyHistory"],
            "work_interfere": request.form["inputWorkInterference"],
            "no_employees": request.form["inputNoOfEmp"],
            "remote_work": request.form["inputRemoteWork"],
            "tech_company": request.form["inputTechCompany"],
            "benefits": request.form["inputBenefits"],
            "care_options": request.form["inputCareOptions"],
            "wellness_program": request.form["inputWellnessProgram"],
            "seek_help": request.form["inputSeekHelp"],
            "anonymity": request.form["inputAnonymity"],
            "leave": request.form["inputLeave"],
            "mental_health_consequence": request.form["inputMentalHealthConsequence"],
            "phys_health_consequence": request.form["inputPhysHealthConsequence"],
            "coworkers": request.form["inputCoworkers"],
            "supervisor": request.form["inputSupervisor"],
            "mental_health_interview": request.form["inputMentalHealthInterview"],
            "phys_health_interview": request.form["inputPhysHealthInterview"],
            "mental_vs_physical": request.form["inputMentalVsPhysical"],
            "obs_consequence": request.form["inputObsConsequence"],
        }]

        df = DataFrame.from_records(data)

        # Transform and predict
        x = ct.transform(df)
        y = model.predict(x)
        treatment = le.inverse_transform(y)[0]

        if treatment == "Yes":
            message = "The prediction result is: Yes. You may have Mental Health Problems. Please visit a nearby psychiatrist."
        else:
            message = "The prediction result is: No. You seem to not have Mental Health Problems."

        return render_template("result.html", name=name, message=message)

    except Exception as e:
        return f"An error occurred: {e}", 500

@app.route('/teampage')
def teampage():
    return 'Welcome to Team page.'

if __name__ == "__main__":
    app.run(debug=False, port=8000)
