from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load dataset
df = pd.read_csv("data/placement_predict_50k Dataset.csv")

# Dashboard metrics
metrics = {
    "total_students": len(df),

    "placed_students": len(df[df["PlacementStatus"] == 1]),

    "not_placed": len(df[df["PlacementStatus"] == 0]),

    "average_salary": round(df["Salary Package"].mean(), 2)
}
# Display first 10 students
STUDENTS_DATA = df.head(10).to_dict(orient="records")


@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        cgpa = float(request.form.get("cgpa", 0))
        internships = int(request.form.get("internships", 0))
        projects = int(request.form.get("projects", 0))
        aptitude = int(request.form.get("aptitude", 0))
        communication = int(request.form.get("communication", 0))
        backlogs = int(request.form.get("backlogs", 0))

        score = (
            (cgpa * 10)
            + (internships * 5)
            + (projects * 4)
            + (aptitude * 0.3)
            + (communication * 4)
            - (backlogs * 12)
        )

        if score >= 100 and backlogs == 0:

            status = "PLACED"

            salary = round(
                10 +
                (cgpa - 9.0) * 1.1 +
                (projects * 0.01),
                2
            )

            placed_prob = min(round(score / 1.3, 1), 98.5)
            not_placed_prob = round(100 - placed_prob, 1)

        else:

            status = "NOT PLACED"
            salary = 0
            placed_prob = 15
            not_placed_prob = 85

        result = {
            "status": status,
            "salary": f"{salary} LPA" if status == "PLACED" else "N/A",
            "placed_prob": placed_prob,
            "not_placed_prob": not_placed_prob
        }

    return render_template(
        "index.html",
        metrics=metrics,
        students=STUDENTS_DATA,
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)