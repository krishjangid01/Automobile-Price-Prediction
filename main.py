import pandas as pd
import joblib


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_PATH = "models/saved_models/linear_regression_model.pkl"
FEATURES_PATH = "dataset/selected_features.csv"


# ==================================================
# LOAD MODEL AND FEATURES
# ==================================================

model = joblib.load(MODEL_PATH)

selected_data = pd.read_csv(FEATURES_PATH)
FEATURE_COLUMNS = selected_data.drop(columns=["price"]).columns.tolist()


# ==================================================
# INPUT HELPER FUNCTIONS
# ==================================================

def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("❌ Please enter a valid number.")


def get_text_input(prompt):
    while True:
        value = input(prompt).strip().lower()

        if value:
            return value

        print("❌ Input cannot be empty.")


# ==================================================
# FEATURE ENGINEERING
# ==================================================

def create_features(data):
    data["average-mpg"] = (
        data["city-mpg"] + data["highway-mpg"]
    ) / 2

    data["power-to-weight"] = (
        data["horsepower"] / data["curb-weight"]
    )

    data["engine-weight-ratio"] = (
        data["engine-size"] / data["curb-weight"]
    )

    data["vehicle-volume"] = (
        data["length"] *
        data["width"] *
        data["height"]
    )

    return data


# ==================================================
# PREDICTION SYSTEM
# ==================================================

def predict_price():

    print("\n" + "=" * 55)
    print("        🚗 AUTOMOBILE PRICE PREDICTION")
    print("=" * 55)

    print("\nEnter automobile details:\n")

    # ---------------- Numerical Features ----------------

    engine_size = get_float_input("Engine Size: ")
    curb_weight = get_float_input("Curb Weight: ")
    horsepower = get_float_input("Horsepower: ")
    width = get_float_input("Width: ")
    highway_mpg = get_float_input("Highway MPG: ")
    city_mpg = get_float_input("City MPG: ")
    length = get_float_input("Length: ")
    wheel_base = get_float_input("Wheel Base: ")
    bore = get_float_input("Bore: ")
    height = get_float_input("Height: ")
    peak_rpm = get_float_input("Peak RPM: ")

    # ---------------- Categorical Features ----------------

    make = get_text_input(
        "Make (e.g. toyota, bmw, audi): "
    )

    drive_wheels = get_text_input(
        "Drive Wheels (fwd/rwd/4wd): "
    )

    fuel_system = get_text_input(
        "Fuel System (mpfi/2bbl/idi/etc.): "
    )

    num_cylinders = get_text_input(
        "Number of Cylinders (four/six/five/etc.): "
    )

    engine_type = get_text_input(
        "Engine Type (ohc/ohcv/dohc/etc.): "
    )

    engine_location = get_text_input(
        "Engine Location (front/rear): "
    )

    body_style = get_text_input(
        "Body Style (sedan/hatchback/convertible/etc.): "
    )

    aspiration = get_text_input(
        "Aspiration (std/turbo): "
    )

    fuel_type = get_text_input(
        "Fuel Type (gas/diesel): "
    )

    # ==================================================
    # CREATE INPUT DATAFRAME
    # ==================================================

    data = pd.DataFrame([{
        "engine-size": engine_size,
        "curb-weight": curb_weight,
        "horsepower": horsepower,
        "width": width,
        "highway-mpg": highway_mpg,
        "city-mpg": city_mpg,
        "length": length,
        "wheel-base": wheel_base,
        "bore": bore,
        "height": height,
        "peak-rpm": peak_rpm,

        "make": make,
        "drive-wheels": drive_wheels,
        "fuel-system": fuel_system,
        "num-of-cylinders": num_cylinders,
        "engine-type": engine_type,
        "engine-location": engine_location,
        "body-style": body_style,
        "aspiration": aspiration,
        "fuel-type": fuel_type
    }])

    # ==================================================
    # FEATURE ENGINEERING
    # ==================================================

    data = create_features(data)

    # ==================================================
    # ONE-HOT ENCODING
    # ==================================================

    data = pd.get_dummies(
        data,
        drop_first=True,
        dtype=int
    )

    # ==================================================
    # FEATURE ALIGNMENT
    # ==================================================

    data = data.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    )

    # ==================================================
    # PREDICTION
    # ==================================================

    prediction = model.predict(data)[0]

    # ==================================================
    # RESULT
    # ==================================================

    print("\n" + "=" * 55)
    print("                 📊 RESULT")
    print("=" * 55)

    print(f"\n💰 Predicted Automobile Price: ${prediction:,.2f}")

    print("\n" + "=" * 55)
    print("          Prediction completed successfully!")
    print("=" * 55)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    try:
        predict_price()

    except KeyboardInterrupt:
        print("\n\n⚠️ Prediction cancelled by user.")

    except Exception as error:
        print("\n❌ Something went wrong.")
        print(f"Error: {error}")