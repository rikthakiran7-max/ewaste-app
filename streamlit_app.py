
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

st.set_page_config(
    page_title="E-Waste Identification System",
    page_icon="♻️",
    layout="centered"
)

@st.cache_resource
def load_models():
    efficientnet = load_model("/content/drive/MyDrive/Eprojects/efficientnet_model.h5")
    inception    = load_model("/content/drive/MyDrive/Eprojects/inception_model.h5")
    return efficientnet, inception

CLASS_NAMES = [
    "Battery", "Keyboard", "Microwave", "Mobile",
    "Mouse", "PCB", "Player", "Printer",
    "Television", "Washing Machine"
]

# ============================================
# ALL DATA
# ============================================

HAZARD_LEVELS = {
    "Battery":         {"level": "HIGH",   "color": "🔴", "reason": "Contains lead, lithium, mercury — toxic and flammable"},
    "PCB":             {"level": "HIGH",   "color": "🔴", "reason": "Contains lead solder, cadmium, brominated flame retardants"},
    "Television":      {"level": "HIGH",   "color": "🔴", "reason": "CRT TVs contain lead; LCDs contain mercury backlights"},
    "Microwave":       {"level": "MEDIUM", "color": "🟡", "reason": "Contains capacitors that can store electric charge"},
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

# Feature 1 — Metal Extraction Guide
METAL_EXTRACTION = {
    "Mobile": [
        {"metal": "Gold", "location": "Circuit board (PCB) pins and connectors", "method": "Acid leaching using nitric acid — done by certified recyclers only"},
        {"metal": "Silver", "location": "Circuit board contacts", "method": "Chemical precipitation process"},
        {"metal": "Copper", "location": "Wiring and circuit board", "method": "Smelting and electrolytic refining"},
        {"metal": "Lithium", "location": "Battery", "method": "Hydrometallurgical process — battery is crushed and chemically processed"},
        {"metal": "Cobalt", "location": "Battery cathode", "method": "Solvent extraction from battery material"},
        {"metal": "Rare Earth (Neodymium)", "location": "Vibration motor", "method": "Magnetic separation and chemical processing"}
    ],
    "PCB": [
        {"metal": "Gold", "location": "Edge connectors, chip pins, contact pads", "method": "Aqua regia (acid) dissolution — highly controlled process"},
        {"metal": "Silver", "location": "Solder joints and surface contacts", "method": "Electrolytic silver recovery"},
        {"metal": "Copper", "location": "Entire board traces and layers", "method": "Smelting followed by electrolytic refining"},
        {"metal": "Palladium", "location": "Capacitors and connectors", "method": "Acid leaching and solvent extraction"},
        {"metal": "Tin", "location": "Solder joints", "method": "Thermal treatment and gravity separation"}
    ],
    "Television": [
        {"metal": "Copper", "location": "Wiring and coils", "method": "Wire stripping and smelting"},
        {"metal": "Aluminum", "location": "Heat sinks and frame", "method": "Mechanical separation and melting"},
        {"metal": "Lead", "location": "CRT glass (old TVs)", "method": "Glass crushing and lead smelting — certified facilities only"},
        {"metal": "Indium", "location": "LCD screen panel", "method": "Chemical leaching of LCD panel"}
    ],
    "Battery": [
        {"metal": "Lithium", "location": "Battery cells (cathode/anode)", "method": "Hydrometallurgical — battery crushed, lithium extracted chemically"},
        {"metal": "Cobalt", "location": "Cathode material", "method": "Solvent extraction and electrowinning"},
        {"metal": "Nickel", "location": "NiMH batteries", "method": "Pyrometallurgical smelting process"},
        {"metal": "Lead", "location": "Lead-acid batteries", "method": "Battery breaking and lead smelting — most recycled metal in world"}
    ],
    "Microwave": [
        {"metal": "Copper", "location": "Magnetron coils and wiring", "method": "Wire stripping and copper smelting"},
        {"metal": "Aluminum", "location": "Outer casing and interior walls", "method": "Mechanical shredding and separation"},
        {"metal": "Steel", "location": "Body and chassis", "method": "Magnetic separation and steel recycling"}
    ],
    "Printer": [
        {"metal": "Copper", "location": "Motor windings and PCB", "method": "Smelting after mechanical separation"},
        {"metal": "Aluminum", "location": "Frame components", "method": "Mechanical separation and melting"},
        {"metal": "Steel", "location": "Chassis and rollers", "method": "Magnetic separation"}
    ],
    "Keyboard": [
        {"metal": "Copper", "location": "PCB traces and connectors", "method": "PCB recycling process"},
        {"metal": "Steel", "location": "Metal plate under keys", "method": "Magnetic separation"}
    ],
    "Mouse": [
        {"metal": "Copper", "location": "Internal wiring and PCB", "method": "Wire stripping and PCB recycling"},
        {"metal": "Steel", "location": "Scroll wheel shaft", "method": "Mechanical separation"}
    ],
    "Player": [
        {"metal": "Gold", "location": "PCB connectors", "method": "Acid leaching by certified recycler"},
        {"metal": "Lithium", "location": "Battery", "method": "Hydrometallurgical battery recycling"},
        {"metal": "Copper", "location": "PCB and wiring", "method": "Smelting process"}
    ],
    "Washing Machine": [
        {"metal": "Steel", "location": "Drum, body and chassis", "method": "Shredding and magnetic separation"},
        {"metal": "Copper", "location": "Motor windings", "method": "Motor dismantling and wire stripping"},
        {"metal": "Aluminum", "location": "Some motor parts", "method": "Eddy current separation"}
    ]
}

# Feature 2 — Hazardous Metals
HAZARDOUS_METALS = {
    "Mobile": [
        {"metal": "Lead", "location": "Solder on circuit board", "health_risk": "Damages brain, kidney and nervous system", "disposal": "Take to certified e-waste center — never burn or crush"},
        {"metal": "Mercury", "location": "LCD backlight (older models)", "health_risk": "Causes neurological damage and kidney failure", "disposal": "Handle with gloves, send to mercury-certified recycler"},
        {"metal": "Cadmium", "location": "Battery (NiCd type)", "health_risk": "Causes cancer and kidney damage", "disposal": "Never landfill — mandatory certified recycling"},
        {"metal": "Arsenic", "location": "Microchips and semiconductors", "health_risk": "Carcinogenic — causes lung and skin cancer", "disposal": "Only handled by licensed hazardous waste facility"}
    ],
    "PCB": [
        {"metal": "Lead", "location": "All solder joints across the board", "health_risk": "Neurotoxin — especially harmful to children", "disposal": "Certified e-waste recycler only — never incinerate"},
        {"metal": "Cadmium", "location": "Chip resistors and semiconductors", "health_risk": "Accumulates in kidneys causing permanent damage", "disposal": "Hazardous waste facility only"},
        {"metal": "Chromium (Hexavalent)", "location": "Metal plating on board", "health_risk": "Carcinogenic — causes DNA damage", "disposal": "Licensed hazardous waste handler only"},
        {"metal": "Beryllium", "location": "Connector springs and relays", "health_risk": "Causes chronic lung disease (berylliosis)", "disposal": "Never grind or sand — certified recycler only"}
    ],
    "Battery": [
        {"metal": "Lead", "location": "Lead plates inside lead-acid battery", "health_risk": "Severe neurotoxin affecting brain development", "disposal": "Return to battery shop — most shops accept old batteries"},
        {"metal": "Cadmium", "location": "NiCd battery electrodes", "health_risk": "Carcinogenic — banned in many countries", "disposal": "Mandatory certified recycling — never landfill"},
        {"metal": "Mercury", "location": "Button cell batteries", "health_risk": "Damages brain and nervous system", "disposal": "Special mercury disposal bin — never crush"},
        {"metal": "Lithium", "location": "Li-ion battery cells", "health_risk": "Fire and explosion risk if damaged", "disposal": "Never puncture — certified battery recycler only"}
    ],
    "Television": [
        {"metal": "Lead", "location": "CRT glass contains 2-4 kg of lead per TV", "health_risk": "Leaches into soil and groundwater", "disposal": "Specialized CRT recycler — never smash the screen"},
        {"metal": "Mercury", "location": "Fluorescent backlight in LCD TVs", "health_risk": "Vapor is highly toxic when inhaled", "disposal": "Intact unit to certified recycler — never break screen"},
        {"metal": "Cadmium", "location": "Phosphor coating in CRT screen", "health_risk": "Carcinogenic when dust is inhaled", "disposal": "CRT-certified recycler only"}
    ],
    "Microwave": [
        {"metal": "Beryllium", "location": "Magnetron tube (microwave generator)", "health_risk": "Causes berylliosis — fatal lung disease", "disposal": "Never drill or cut magnetron — certified recycler only"},
        {"metal": "Lead", "location": "Solder in control PCB", "health_risk": "Neurotoxin", "disposal": "PCB to certified e-waste handler"}
    ],
    "Printer": [
        {"metal": "Lead", "location": "Solder on main PCB", "health_risk": "Neurotoxin", "disposal": "Certified e-waste recycler"},
        {"metal": "Cadmium", "location": "Some toner powders (older printers)", "health_risk": "Carcinogenic when inhaled as dust", "disposal": "Sealed disposal — never vacuum loose toner"}
    ],
    "Keyboard": [
        {"metal": "Lead", "location": "Solder on PCB", "health_risk": "Low risk if intact — risk when dismantled", "disposal": "E-waste bin or certified recycler"}
    ],
    "Mouse": [
        {"metal": "Lead", "location": "Solder on small PCB", "health_risk": "Low risk if intact", "disposal": "E-waste bin or certified recycler"}
    ],
    "Player": [
        {"metal": "Lead", "location": "PCB solder joints", "health_risk": "Neurotoxin", "disposal": "Certified e-waste center"},
        {"metal": "Lithium", "location": "Rechargeable battery", "health_risk": "Fire risk if punctured", "disposal": "Battery recycling point"}
    ],
    "Washing Machine": [
        {"metal": "Lead", "location": "Motor PCB solder", "health_risk": "Low risk — mostly safe", "disposal": "Authorized scrap dealer or e-waste center"}
    ]
}

# Feature 3 — Certified Buyers
CERTIFIED_BUYERS = [
    {"name": "E-Parisaraa Pvt Ltd", "location": "Bangalore, Karnataka", "contact": "+91-80-28372373", "website": "www.e-parisaraa.com", "accepts": "All e-waste categories"},
    {"name": "Attero Recycling", "location": "Noida, UP (Pan India pickup)", "contact": "+91-120-4781000", "website": "www.attero.in", "accepts": "Mobile, Battery, PCB, Laptop"},
    {"name": "Eco Recycling Ltd (Ecoreco)", "location": "Mumbai, Maharashtra", "contact": "+91-22-40064800", "website": "www.ecoreco.com", "accepts": "All e-waste categories"},
    {"name": "Karo Sambhav", "location": "Pan India", "contact": "+91-11-41012592", "website": "www.karosambhav.com", "accepts": "Mobile, TV, Appliances"},
    {"name": "Greenscape Eco Management", "location": "Bangalore, Karnataka", "contact": "+91-80-23490050", "website": "www.greenscapeindia.com", "accepts": "All e-waste categories"},
    {"name": "TES-AMM India", "location": "Bangalore, Karnataka", "contact": "+91-80-67086000", "website": "www.tes-amm.com", "accepts": "IT equipment, PCB, Mobile"},
    {"name": "Namo e-Waste Management", "location": "Nashik, Maharashtra", "contact": "+91-253-2317100", "website": "www.namoemgmt.com", "accepts": "All e-waste categories"},
    {"name": "Clean Harbors India", "location": "Pan India", "contact": "+91-22-61923000", "website": "www.cleanharbors.com", "accepts": "Hazardous e-waste, Battery"}
]

# Feature 4 — CO2 Impact
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

# Feature 5 — Disposal Centers by City
DISPOSAL_CENTERS = {
    "bangalore": [
        {"name": "E-Parisaraa", "address": "No 8, 3rd Cross, Peenya Industrial Area, Bangalore", "phone": "080-28372373", "maps": "https://maps.google.com/?q=E-Parisaraa+Peenya+Bangalore"},
        {"name": "Greenscape Eco", "address": "Bommasandra Industrial Area, Bangalore", "phone": "080-23490050", "maps": "https://maps.google.com/?q=Greenscape+Eco+Bommasandra+Bangalore"},
        {"name": "TES-AMM", "address": "Plot 38, KIADB Industrial Area, Bangalore", "phone": "080-67086000", "maps": "https://maps.google.com/?q=TES+AMM+KIADB+Bangalore"}
    ],
    "mangalore": [
        {"name": "KSPCB E-Waste Collection Center", "address": "KSPCB Regional Office, Kodialbail, Mangalore", "phone": "0824-2452304", "maps": "https://maps.google.com/?q=KSPCB+Kodialbail+Mangalore"},
        {"name": "Mangalore City Corporation E-Waste", "address": "MCC Main Office, Lalbagh, Mangalore", "phone": "0824-2220283", "maps": "https://maps.google.com/?q=Mangalore+City+Corporation+Lalbagh"},
        {"name": "Attero Drop Point — Mangalore", "address": "Hampankatta, Mangalore", "phone": "1800-102-7070", "maps": "https://maps.google.com/?q=Hampankatta+Mangalore"}
    ],
    "mumbai": [
        {"name": "Eco Recycling Ltd", "address": "Andheri East, Mumbai", "phone": "022-40064800", "maps": "https://maps.google.com/?q=Eco+Recycling+Andheri+Mumbai"},
        {"name": "Clean Harbors", "address": "MIDC Turbhe, Navi Mumbai", "phone": "022-61923000", "maps": "https://maps.google.com/?q=Clean+Harbors+MIDC+Navi+Mumbai"}
    ],
    "delhi": [
        {"name": "Attero Recycling HQ", "address": "Sector 63, Noida, Delhi NCR", "phone": "120-4781000", "maps": "https://maps.google.com/?q=Attero+Recycling+Sector+63+Noida"},
        {"name": "Karo Sambhav Center", "address": "Okhla Industrial Area, New Delhi", "phone": "011-41012592", "maps": "https://maps.google.com/?q=Karo+Sambhav+Okhla+Delhi"}
    ],
    "chennai": [
        {"name": "E-Waste Recyclers India", "address": "Ambattur Industrial Estate, Chennai", "phone": "044-26583636", "maps": "https://maps.google.com/?q=Ambattur+Industrial+Estate+Chennai"},
        {"name": "Ramky E-Waste", "address": "Guindy Industrial Area, Chennai", "phone": "044-22501234", "maps": "https://maps.google.com/?q=Guindy+Industrial+Area+Chennai"}
    ],
    "hyderabad": [
        {"name": "E-Waste Recycling Hub", "address": "IDA Nacharam, Hyderabad", "phone": "040-27152345", "maps": "https://maps.google.com/?q=IDA+Nacharam+Hyderabad"},
        {"name": "Greentech e-Waste", "address": "Patancheru Industrial Area, Hyderabad", "phone": "040-23090123", "maps": "https://maps.google.com/?q=Patancheru+Industrial+Area+Hyderabad"}
    ]
}

# ============================================
# PREDICTION FUNCTION
# ============================================
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

# ============================================
# MAIN APP
# ============================================
def main():
    st.title("♻️ AI-Powered E-Waste Identification System")
    st.markdown("### Identify • Classify • Recommend • Recycle")
    st.markdown("---")

    with st.spinner("🔄 Loading AI models..."):
        efficientnet_model, inception_model = load_models()
    st.success("✅ Hybrid AI Model loaded! (EfficientNetB0 + InceptionV3 — 97% accuracy)")

    st.markdown("## 📤 Upload E-Waste Image")
    uploaded_file = st.file_uploader(
        "Choose an image of e-waste...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Identify E-Waste", type="primary"):
            with st.spinner("🤖 Analyzing image using Hybrid AI Model..."):
                device, confidence = predict(img, efficientnet_model, inception_model)

            st.markdown("---")

            # ── Section 1: Detection Result ──
            st.markdown("## 🎯 Detection Result")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Device Detected", device)
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
            with col3:
                st.metric("Model Used", "Hybrid AI")

            st.markdown("---")

            # ── Section 2: Hazard Level ──
            st.markdown("## ☠️ Hazard Classification")
            hazard       = HAZARD_LEVELS[device]
            hazard_color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}[hazard["level"]]
            st.markdown(
                f"<h3 style=color:{hazard_color}>{hazard["color"]} {hazard["level"]} RISK</h3>",
                unsafe_allow_html=True)
            st.info(f"**Reason:** {hazard["reason"]}")

            st.markdown("---")

            # ── Section 3: Scrap Value ──
            st.markdown("## 💰 Estimated Scrap Value")
            value = VALUE_DATA[device]
            st.metric("Estimated Value", f"₹{value}")
            st.caption("Note: Actual value depends on device condition and current market rates")

            st.markdown("---")

            # ── Section 4: Metal Extraction ──
            st.markdown("## 🔩 Recoverable Metals & Extraction Methods")
            st.info("These are the valuable metals that can be recovered from your device by certified recyclers")
            metals = METAL_EXTRACTION.get(device, [])
            for i, m in enumerate(metals):
                with st.expander(f"⚙️ {m["metal"]} — Found in: {m["location"]}"):
                    st.write(f"**📍 Location:** {m["location"]}")
                    st.write(f"**🔧 Extraction Method:** {m["method"]}")

            st.markdown("---")

            # ── Section 5: Hazardous Metals ──
            st.markdown("## ⚠️ Hazardous Materials — Safe Disposal Guide")
            st.warning("These materials require special handling. Never throw in regular trash!")
            haz_metals = HAZARDOUS_METALS.get(device, [])
            for m in haz_metals:
                with st.expander(f"☠️ {m["metal"]} — Location: {m["location"]}"):
                    st.write(f"**📍 Location in Device:** {m["location"]}")
                    st.write(f"**🩺 Health Risk:** {m["health_risk"]}")
                    st.write(f"**✅ Safe Disposal:** {m["disposal"]}")

            st.markdown("---")

            # ── Section 6: CO2 Impact ──
            st.markdown("## 🌍 Environmental Impact of Recycling")
            co2 = CO2_IMPACT.get(device, {"co2_saved": 0, "trees": 0, "fact": ""})
            col1, col2 = st.columns(2)
            with col1:
                st.metric("CO2 Saved by Recycling", f"{co2["co2_saved"]} kg")
            with col2:
                st.metric("Equivalent Trees Planted", f"🌳 {co2["trees"]} trees")
            st.success(f"💡 Did you know? {co2["fact"]}")

            st.markdown("---")

            # ── Section 7: Recycling Recommendations ──
            st.markdown("## ♻️ Recycling Recommendations")
            for rec in RECOMMENDATIONS[device]:
                st.markdown(f"- {rec}")

            st.markdown("---")

            # ── Section 8: Certified Buyers ──
            st.markdown("## 🏪 Certified E-Waste Buyers in India")
            st.info("These are government authorized e-waste recyclers. Always sell to certified buyers only!")
            for buyer in CERTIFIED_BUYERS:
                with st.expander(f"🏭 {buyer["name"]} — {buyer["location"]}"):
                    st.write(f"**📞 Contact:** {buyer["contact"]}")
                    st.write(f"**🌐 Website:** {buyer["website"]}")
                    st.write(f"**✅ Accepts:** {buyer["accepts"]}")

            st.markdown("---")

            # ── Section 9: Disposal Finder ──
            st.markdown("## 📍 Find E-Waste Disposal Center Near You")
            city = st.text_input(
                "Enter your city name:",
                placeholder="e.g. Bangalore, Mangalore, Mumbai, Delhi, Chennai, Hyderabad"
            )
            if city:
                city_key = city.lower().strip()
                centers  = DISPOSAL_CENTERS.get(city_key, [])
                if centers:
                    st.success(f"✅ Found {len(centers)} disposal centers in {city.title()}!")
                    for center in centers:
                        with st.expander(f"📍 {center["name"]}"):
                            st.write(f"**📌 Address:** {center["address"]}")
                            st.write(f"**📞 Phone:** {center["phone"]}")
                            st.markdown(f"[🗺️ Open in Google Maps]({center["maps"]})")
                else:
                    st.warning(f"No centers found for {city.title()} yet.")
                    st.info("Try: Bangalore, Mangalore, Mumbai, Delhi, Chennai or Hyderabad")
                    st.markdown("🌐 Or visit [CPCB E-Waste Portal](https://cpcb.nic.in) to find centers near you")

            st.markdown("---")
            st.success("✅ Analysis Complete! Please recycle responsibly. 🌍")

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style=text-align:center;color:gray>♻️ AI-Powered E-Waste System | Srinivas Institute of Technology | Riktha Kiran & Prajwal Kunder</p>",
        unsafe_allow_html=True)

if __name__ == "__main__":
    main()
