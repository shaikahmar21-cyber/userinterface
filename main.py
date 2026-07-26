from flask import Flask, render_template, request, redirect

app = Flask(__name__)

students = []

@app.route("/")
def home():
    return render_template("index.html", students=students)

@app.route("/add", methods=["POST"])
def add_student():
    name = request.form["name"]
    branch = request.form["branch"]
    year = request.form["year"]

    students.append({
        "name": name,
        "branch": branch,
        "year": year
    })

    return redirect("/")

@app.route("/delete/<int:index>")
def delete(index):
    if 0 <= index < len(students):
        students.pop(index)
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)