from flask import Flask, render_template, request

from recommender import advisor_recommendation


app = Flask(__name__)


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():

    purpose = request.form["purpose"]

    budget = int(request.form["budget"])

    ram = int(request.form["ram"])

    priority = request.form["priority"]


    laptops = advisor_recommendation(
        purpose,
        budget,
        ram,
        priority
    )


    return render_template(
        "result.html",
        laptops=laptops,
        purpose=purpose,
        priority=priority
    )


if __name__ == "__main__":
    app.run(debug=True)