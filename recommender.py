import joblib


# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

knn = joblib.load("knn_model.pkl")

saved = joblib.load("encoders.pkl")


core_encoder = saved["core"]

gpu_encoder = saved["gpu"]

os_encoder = saved["os"]

scaler = saved["scaler"]

df = saved["data"]



# ------------------------------------------------
# MAIN RECOMMENDATION FUNCTION
# ------------------------------------------------

def get_recommendations(
        budget,
        ram,
        core,
        graphics,
        purpose
):


    # Encode processor

    try:
        core_value = core_encoder.transform([core])[0]

    except:
        core_value = 0



    # Encode graphics

    try:
        gpu_value = gpu_encoder.transform([graphics])[0]

    except:
        gpu_value = 0



    # Default OS

    os_value = df["OSEncoded"].mode()[0]


    # Average rating

    rating = df["Rating"].mean()



    sample = [[
        budget,
        ram,
        rating,
        core_value,
        gpu_value,
        os_value
    ]]


    # Scale input exactly like training

    sample = scaler.transform(sample)



    # Find nearest laptops

    distances, indices = knn.kneighbors(sample)



    recommendations = []



    for index, distance in zip(
            indices[0],
            distances[0]
    ):


        laptop = df.iloc[index]



        # Convert distance into percentage

        match_score = int(
            100 - (distance * 20)
        )



        match_score = max(
            50,
            min(match_score, 99)
        )



        recommendations.append({

            "Model": laptop["Model"],

            "Price": laptop["Price"],

            "Rating": laptop["Rating"],

            "Generation": laptop["Generation"],

            "Core": laptop["Core"],

            "Ram": laptop["Ram"],

            "SSD": laptop["SSD"],

            "Display": laptop["Display"],

            "Graphics": laptop["Graphics"],

            "OS": laptop["OS"],

            "Warranty": laptop["Warranty"],

            "match": match_score,

            "reason": generate_reason(
                purpose,
                laptop
            )

        })



    return recommendations





# ------------------------------------------------
# AI ADVISOR
# ------------------------------------------------

def advisor_recommendation(
        purpose,
        budget,
        ram,
        priority
):


    core = choose_processor("medium")

    graphics = choose_gpu()



    if purpose == "Gaming":

        core = choose_processor("high")

        graphics = choose_gpu()



    elif purpose == "Editing":

        core = choose_processor("high")

        graphics = choose_gpu()



    elif purpose == "Coding":

        core = choose_processor("medium")



    elif purpose == "Student":

        core = choose_processor("low")





    return get_recommendations(

        budget,

        ram,

        core,

        graphics,

        purpose

    )





# ------------------------------------------------
# PROCESSOR SELECTOR
# ------------------------------------------------

def choose_processor(level):


    processors = df["Core"].astype(str).tolist()



    for cpu in processors:


        text = cpu.lower()



        if level == "high":

            if (
                "i7" in text
                or
                "i9" in text
                or
                "ryzen 7" in text
            ):
                return cpu



        elif level == "medium":

            if (
                "i5" in text
                or
                "ryzen 5" in text
            ):
                return cpu



        elif level == "low":

            return cpu



    return processors[0]





# ------------------------------------------------
# GPU SELECTOR
# ------------------------------------------------

def choose_gpu():


    graphics = df["Graphics"].astype(str)



    for gpu in graphics:


        text = gpu.lower()



        if (
            "rtx" in text
            or
            "gtx" in text
            or
            "radeon" in text
        ):

            return gpu



    return graphics.iloc[0]





# ------------------------------------------------
# REASON GENERATOR
# ------------------------------------------------

def generate_reason(
        purpose,
        laptop
):


    reasons = []


    reasons.append(
        f"Optimized for {purpose} usage"
    )


    reasons.append(
        f"Processor: {laptop['Core']}"
    )


    reasons.append(
        f"Graphics: {laptop['Graphics']}"
    )


    reasons.append(
        "Features matched using KNN similarity"
    )


    return reasons





def get_core_options():

    return sorted(
        df["Core"].dropna().unique().tolist()
    )





def get_graphics_options():

    return sorted(
        df["Graphics"].dropna().unique().tolist()
    )