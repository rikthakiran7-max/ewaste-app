
import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

st.set_page_config(
    page_title="E-Waste Identification System",
    page_icon="♻️",
    layout="centered"
)

@st.cache_resource
def load_models():
    # Download models from Google Drive if not present
    if not os.path.exists("efficientnet_model.h5"):
        with st.spinner("📥 Downloading EfficientNetB0 model..."):
            gdown.download(
                "https://drive.google.com/uc?id=1xjwr5GQX6dN4NFIlKUzXQWMYFcNKf0FT",
                "efficientnet_model.h5",
                quiet=False
            )

    if not os.path.exists("inception_model.h5"):
        with st.spinner("📥 Downloading InceptionV3 model..."):
            gdown.download(
                "https://drive.google.com/uc?id=1mGqUktqK3MK_KVmS9xqtcvM-ezBUuf6D",
                "inception_model.h5",
                quiet=False
            )

    efficientnet = load_model("efficientnet_model.h5")
    inception    = load_model("inception_model.h5")
    return efficientnet, inception

CLASS_NAMES = [
    "Battery", "Keyboard", "Microwave", "Mobile",
    "Mouse", "PCB", "Player", "Printer",
    "Television", "Washing Machine"
]

HAZARD_LEVELS = {
    "Battery":         {"level": "HIGH",   "color": "🔴", "reason": "Contains lead, lithium, mercury — toxic and flammable"},
    "PCB":             {"level": "HIGH",   "color": "🔴", "reason": "Contains lead solder, cadmium, brominated flame retardants"},
    "Television":      {"level": "HIGH",   "color": "🔴", "reason": "CRT TVs contain lead; LCDs contain mercury backlights"},
    "Microwave":       {"level": "MEDIUM", "color": "🟡", "reason": "Contains capacitors that store electric charge"},
    "Printer":         {"level": "MEDIUM", "color": "🟡", "reason": "Contains toner particles and ink chemicals"},
    "Mobile":          {"level": "MEDIUM", "color": "🟡", "reason": "Contains lithium battery and rare earth metals"},
    "Player":          {"level": "MEDIUM", "color": "🟡", "reason": "Contains battery and circuit boards with heavy metals"},
    "Washing Machine": {"level": "LOW",    "color": "🟢", "reason": "Mostly metal and plastic, low toxic content"},
    "Keyboard":        {"level": "LOW",    "color": "🟢", "reason": "Mostly plastic, minimal hazardous materials"},
    "Mouse":           {"level": "LOW",    "color": "🟢", "reason": "Mostly plastic and copper wire, low risk"}
}

VALUE_DATA = {
    "Battery": 50,   "Keyboard": 30,  "Microwave": 200,
    "Mobile": 150,   "Mouse": 20,     "PCB": 500,
    "Player": 80,    "Printer": 120,  "Television": 300,
    "Washing Machine": 400
}

RECOMMENDATIONS = {
    "Battery":         ["♻️ Drop at certified battery collection points", "🏭 Send to authorized battery recycler", "⚠️ NEVER dispose in regular trash or burn"],
    "PCB":             ["♻️ Send to e-waste recycler for precious metal recovery", "🏭 Contact certified e-waste handler", "⚠️ Do not dismantle without protective equipment"],
    "Mobile":          ["📱 Trade-in or donate if still working", "♻️ Drop at brand collection drives", "🏭 Send to e-waste recycler for battery recovery"],
    "Television":      ["♻️ Contact municipal e-waste collection program", "🏭 Certified recycler for CRT screens", "⚠️ Do not break CRT screens — contains lead"],
    "Keyboard":        ["♻️ Donate if functional", "🗑️ Drop at e-waste bin", "🔧 Plastic parts to recycler"],
    "Mouse":           ["♻️ Drop at e-waste bin", "🔧 Copper wire can be recovered", "🗑️ Plastic body to recycler"],
    "Printer":         ["♻️ Return cartridges to manufacturer", "🏭 Send to certified e-waste recycler", "⚠️ Handle toner carefully"],
    "Microwave":       ["⚡ Discharge capacitors before handling", "♻️ Metal parts to scrap dealer", "🏭 Circuit board to e-waste recycler"],
    "Player":          ["♻️ Drop at e-waste collection point", "🔧 Remove battery separately", "🏭 Send to certified e-waste recycler"],
    "Washing Machine": ["🔧 Metal drum to scrap dealer", "♻️ Plastic parts to recycler", "🏭 Motor and PCB to e-waste recycler"]
}

CO2_IMPACT = {
    "Mobile":          {"co2_saved": 70,  "trees": 3,  "fact": "Making one new phone generates 70kg CO2. Recycling saves all of it!"},
    "PCB":             {"co2_saved": 45,  "trees": 2,  "fact": "PCB recycling recovers gold without mining — saving massive CO2 emissions"},
    "Television":      {"co2_saved": 150, "trees": 7,  "fact": "Recycling one TV saves enough energy to power a home for 3 weeks"},
    "Battery":         {"co2_saved": 30,  "trees": 1,  "fact": "Lithium battery recycling reduces need for new lithium mining by 90%"},
    "Microwave":       {"co2_saved": 80,  "trees": 4,  "fact": "Recycling microwave steel and copper saves 80kg of CO2 emissions"},
    "Printer":         {"co2_saved": 55,  "trees": 2,  "fact": "Cartridge recycling alone saves 2.5kg CO2 per cartridge"},
    "Keyboard":        {"co2_saved": 15,  "trees": 1,  "fact": "Keyboard recycling recovers plastic that would take 500 years to decompose"},
    "Mouse":           {"co2_saved": 10,  "trees": 1,  "fact": "Even small devices matter — copper in mouse takes high energy to mine fresh"},
    "Player":          {"co2_saved": 25,  "trees": 1,  "fact": "Recycling portable players recovers rare earth metals used in motors"},
    "Washing Machine": {"co2_saved": 200, "trees": 9,  "fact": "Washing machine has 25kg of steel — recycling steel saves 1.5 tonnes of CO2 per tonne"}
}

CERTIFIED_BUYERS = [
    {"name": "E-Parisaraa Pvt Ltd", "location": "Bangalore, Karnataka", "contact": "+91-80-28372373", "website": "www.e-parisaraa.com", "accepts": "All e-waste categories"},
    {"name": "Attero Recycling", "location": "Noida, UP (Pan India pickup)", "contact": "+91-120-4781000", "website": "www.attero.in", "accepts": "Mobile, Battery, PCB, Laptop"},
    {"name": "Eco Recycling Ltd", "location": "Mumbai, Maharashtra", "contact": "+91-22-40064800", "website": "www.ecoreco.com", "accepts": "All e-waste categories"},
    {"name": "Karo Sambhav", "location": "Pan India", "contact": "+91-11-41012592", "website": "www.karosambhav.com", "accepts": "Mobile, TV, Appliances"},
    {"name": "Greenscape Eco", "location": "Bangalore, Karnataka", "contact": "+91-80-23490050", "website": "www.greenscapeindia.com", "accepts": "All e-waste categories"}
]

DISPOSAL_CENTERS = {
    "bangalore": [
        {"name": "E-Parisaraa", "address": "Peenya Industrial Area, Bangalore", "phone": "080-28372373", "maps": "https://maps.google.com/?q=E-Parisaraa+Peenya+Bangalore"},
        {"name": "Greenscape Eco", "address": "Bommasandra Industrial Area, Bangalore", "phone": "080-23490050", "maps": "https://maps.google.com/?q=Greenscape+Eco+Bommasandra+Bangalore"}
    ],
    "mangalore": [
        {"name": "KSPCB E-Waste Center", "address": "Kodialbail, Mangalore", "phone": "0824-2452304", "maps": "https://maps.google.com/?q=KSPCB+Kodialbail+Mangalore"},
        {"name": "MCC E-Waste", "address": "Lalbagh, Mangalore", "phone": "0824-2220283", "maps": "https://maps.google.com/?q=Mangalore+City+Corporation+Lalbagh"}
    ],
    "mumbai": [
        {"name": "Eco Recycling Ltd", "address": "Andheri East, Mumbai", "phone": "022-40064800", "maps": "https://maps.google.com/?q=Eco+Recycling+Andheri+Mumbai"}
    ],
    "delhi": [
        {"name": "Attero Recycling", "address": "Sector 63, Noida", "phone": "120-4781000", "maps": "https://maps.google.com/?q=Attero+Recycling+Sector+63+Noida"}
    ],
    "chennai": [
        {"name": "E-Waste Recyclers India", "address": "Ambattur Industrial Estate, Chennai", "phone": "044-26583636", "maps": "https://maps.google.com/?q=Ambattur+Industrial+Estate+Chennai"}
    ],
    "hyderabad": [
        {"name": "E-Waste Recycling Hub", "address": "IDA Nacharam, Hyderabad", "phone": "040-27152345", "maps": "https://maps.google.com/?q=IDA+Nacharam+Hyderabad"}
    ]
}

METAL_EXTRACTION = {
    "Mobile": [
        {"metal": "Gold", "location": "Circuit board pins", "method": "Acid leaching by certified recyclers"},
        {"metal": "Lithium", "location": "Battery", "method": "Hydrometallurgical process"},
        {"metal": "Copper", "location": "Wiring and PCB", "method": "Smelting and electrolytic refining"}
    ],
    "PCB": [
        {"metal": "Gold", "location": "Edge connectors and chip pins", "method": "Aqua regia dissolution"},
        {"metal": "Copper", "location": "Board traces", "method": "Smelting and electrolytic refining"},
        {"metal": "Silver", "location": "Solder joints", "method": "Electrolytic silver recovery"}
    ],
    "Battery": [
        {"metal": "Lithium", "location": "Battery cells", "method": "Hydrometallurgical process"},
        {"metal": "Cobalt", "location": "Cathode material", "method": "Solvent extraction"},
        {"metal": "Lead", "location": "Lead-acid batteries", "method": "Lead smelting"}
    ],
    "Television": [
        {"metal": "Copper", "location": "Wiring and coils", "method": "Wire stripping and smelting"},
        {"metal": "Lead", "location": "CRT glass", "method": "Certified CRT recycler only"},
        {"metal": "Aluminum", "location": "Heat sinks", "method": "Mechanical separation"}
    ],
    "Microwave": [
        {"metal": "Copper", "location": "Magnetron coils", "method": "Wire stripping"},
        {"metal": "Steel", "location": "Body and chassis", "method": "Magnetic separation"}
    ],
    "Printer": [
        {"metal": "Copper", "location": "Motor and PCB", "method": "Smelting"},
        {"metal": "Steel", "location": "Chassis", "method": "Magnetic separation"}
    ],
    "Keyboard": [
        {"metal": "Copper", "location": "PCB traces", "method": "PCB recycling"},
        {"metal": "Steel", "location": "Metal plate", "method": "Magnetic separation"}
    ],
    "Mouse": [
        {"metal": "Copper", "location": "Wiring and PCB", "method": "Wire stripping"},
        {"metal": "Steel", "location": "Scroll wheel", "method": "Mechanical separation"}
    ],
    "Player": [
        {"metal": "Gold", "location": "PCB connectors", "method": "Acid leaching"},
        {"metal": "Lithium", "location": "Battery", "method": "Battery recycling"}
    ],
    "Washing Machine": [
        {"metal": "Steel", "location": "Drum and body", "method": "Shredding and magnetic separation"},
        {"metal": "Copper", "location": "Motor windings", "method": "Motor dismantling"}
    ]
}

HAZARDOUS_METALS = {
    "Mobile": [
        {"metal": "Lead", "location": "Solder on circuit board", "health_risk": "Damages brain and nervous system", "disposal": "Certified e-waste center only"},
        {"metal": "Mercury", "location": "LCD backlight", "health_risk": "Neurological damage", "disposal": "Mercury-certified recycler"}
    ],
    "PCB": [
        {"metal": "Lead", "location": "All solder joints", "health_risk": "Neurotoxin harmful to children", "disposal": "Never incinerate — certified recycler only"},
        {"metal": "Cadmium", "location": "Chip resistors", "health_risk": "Accumulates in kidneys", "disposal": "Hazardous waste facility only"}
    ],
    "Battery": [
        {"metal": "Lead", "location": "Lead plates inside battery", "health_risk": "Severe neurotoxin", "disposal": "Return to battery shop"},
        {"metal": "Lithium", "location": "Battery cells", "health_risk": "Fire and explosion risk", "disposal": "Never puncture — certified recycler"}
    ],
    "Television": [
        {"metal": "Lead", "location": "CRT glass — 2 to 4 kg per TV", "health_risk": "Leaches into soil", "disposal": "Specialized CRT recycler only"},
        {"metal": "Mercury", "location": "LCD backlight", "health_risk": "Toxic vapor when inhaled", "disposal": "Never break screen"}
    ],
    "Microwave": [
        {"metal": "Beryllium", "location": "Magnetron tube", "health_risk": "Fatal lung disease", "disposal": "Never drill — certified recycler"}
    ],
    "Printer": [
        {"metal": "Cadmium", "location": "Toner powder", "health_risk": "Carcinogenic when inhaled", "disposal": "Never vacuum loose toner"}
    ],
    "Keyboard": [{"metal": "Lead", "location": "PCB solder", "health_risk": "Low risk if intact", "disposal": "E-waste bin"}],
    "Mouse":    [{"metal": "Lead", "location": "PCB solder", "health_risk": "Low risk if intact", "disposal": "E-waste bin"}],
    "Player":   [{"metal": "Lithium", "location": "Battery", "health_risk": "Fire risk if punctured", "disposal": "Battery recycling point"}],
    "Washing Machine": [{"metal": "Lead", "location": "Motor PCB", "health_risk": "Low risk", "disposal": "Authorized scrap dealer"}]
}

def predict(img, efficientnet_model, inception_model):
    img_224 = img.resize((224, 224))
    arr_224 = keras_image.img_to_array(img_224)
    arr_224 = np.expand_dims(arr_224, axis=0)
    img_299 = img.resize((299, 299))
    arr_299 = keras_image.img_to_array(img_299) / 255.0
    arr_299 = np.expand_dims(arr_299, axis=0)
    eff_pred    = efficientnet_model.predict(arr_224, verbose=0)
    inc_pred    = inception_model.predict(arr_299,    verbose=0)
    hybrid_pred = (eff_pred + inc_pred) / 2
    predicted_idx = np.argmax(hybrid_pred[0])
    confidence    = hybrid_pred[0][predicted_idx] * 100
    device        = CLASS_NAMES[predicted_idx]
    return device, confidence

def main():
    st.title("♻️ AI-Powered E-Waste Identification System")
    st.markdown("### Identify • Classify • Recommend • Recycle")
    st.markdown("---")

    with st.spinner("🔄 Loading AI models — this may take a few minutes on first load..."):
        efficientnet_model, inception_model = load_models()
    st.success("✅ Hybrid AI Model loaded! (EfficientNetB0 + InceptionV3 — 97% accuracy)")

    st.markdown("## 📤 Upload E-Waste Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Identify E-Waste", type="primary"):
            with st.spinner("🤖 Analyzing image..."):
                device, confidence = predict(img, efficientnet_model, inception_model)

            st.markdown("---")
            st.markdown("## 🎯 Detection Result")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Device Detected", device)
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
            with col3:
                st.metric("Model", "Hybrid AI")

            st.markdown("---")
            st.markdown("## ☠️ Hazard Classification")
            hazard = HAZARD_LEVELS[device]
            hazard_color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}[hazard["level"]]
            st.markdown(f"<h3 style=color:{hazard_color}>{hazard["color"]} {hazard["level"]} RISK</h3>", unsafe_allow_html=True)
            st.info(f"**Reason:** {hazard["reason"]}")

            st.markdown("---")
            st.markdown("## 💰 Estimated Scrap Value")
            st.metric("Estimated Value", f"₹{VALUE_DATA[device]}")

            st.markdown("---")
            st.markdown("## 🔩 Recoverable Metals & Extraction Methods")
            for m in METAL_EXTRACTION.get(device, []):
                with st.expander(f"⚙️ {m["metal"]} — {m["location"]}"):
                    st.write(f"**📍 Location:** {m["location"]}")
                    st.write(f"**🔧 Method:** {m["method"]}")

            st.markdown("---")
            st.markdown("## ⚠️ Hazardous Materials — Safe Disposal")
            st.warning("These materials need special handling — never throw in regular trash!")
            for m in HAZARDOUS_METALS.get(device, []):
                with st.expander(f"☠️ {m["metal"]} — {m["location"]}"):
                    st.write(f"**📍 Location:** {m["location"]}")
                    st.write(f"**🩺 Health Risk:** {m["health_risk"]}")
                    st.write(f"**✅ Safe Disposal:** {m["disposal"]}")

            st.markdown("---")
            st.markdown("## 🌍 Environmental Impact")
            co2 = CO2_IMPACT.get(device, {"co2_saved": 0, "trees": 0, "fact": ""})
            col1, col2 = st.columns(2)
            with col1:
                st.metric("CO2 Saved", f"{co2["co2_saved"]} kg")
            with col2:
                st.metric("Equivalent Trees", f"🌳 {co2["trees"]}")
            st.success(f"💡 {co2["fact"]}")

            st.markdown("---")
            st.markdown("## ♻️ Recycling Recommendations")
            for rec in RECOMMENDATIONS[device]:
                st.markdown(f"- {rec}")

            st.markdown("---")
            st.markdown("## 🏪 Certified E-Waste Buyers")
            for buyer in CERTIFIED_BUYERS:
                with st.expander(f"🏭 {buyer["name"]} — {buyer["location"]}"):
                    st.write(f"**📞 Contact:** {buyer["contact"]}")
                    st.write(f"**🌐 Website:** {buyer["website"]}")
                    st.write(f"**✅ Accepts:** {buyer["accepts"]}")

            st.markdown("---")
            st.markdown("## 📍 Find Disposal Center Near You")
            city = st.text_input("Enter your city:", placeholder="Bangalore, Mangalore, Mumbai, Delhi, Chennai, Hyderabad")
            if city:
                centers = DISPOSAL_CENTERS.get(city.lower().strip(), [])
                if centers:
                    st.success(f"✅ Found {len(centers)} centers in {city.title()}!")
                    for c in centers:
                        with st.expander(f"📍 {c["name"]}"):
                            st.write(f"**📌 Address:** {c["address"]}")
                            st.write(f"**📞 Phone:** {c["phone"]}")
                            st.markdown(f"[🗺️ Open in Google Maps]({c["maps"]})")
                else:
                    st.warning("City not found. Try: Bangalore, Mangalore, Mumbai, Delhi, Chennai or Hyderabad")
                    st.markdown("🌐 Or visit [CPCB Portal](https://cpcb.nic.in)")

            st.markdown("---")
            st.success("✅ Analysis Complete! Please recycle responsibly. 🌍")

    st.markdown("---")
    st.markdown("<p style=text-align:center;color:gray>♻️ AI-Powered E-Waste System | Srinivas Institute of Technology | Riktha Kiran & Prajwal Kunder</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
