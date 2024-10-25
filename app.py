import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import pandas as pd


# Mengatur konfigurasi halaman Streamlit
st.set_page_config(
    page_title="SkinPath", 
    page_icon="👩‍⚕️",  
)

# Memuat arsitektur model (ResNet50)
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(in_features=2048, out_features=3)  # Adjust for skin types

# Memuat bobot model
state_dict = torch.load("./model/model.pth")
del state_dict['fc.weight']
del state_dict['fc.bias']
model.load_state_dict(state_dict, strict=False)
model.eval()

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Mengubah ukuran gambar
        transforms.ToTensor(),  # Mengonversi gambar ke tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisasi
    ])
    image = Image.open(io.BytesIO(image))  # Membaca gambar dari byte
    return transform(image).unsqueeze(0)  # Mengembalikan gambar dalam format yang diharapkan model

# Fungsi untuk memprediksi tipe kulit
def predict_skin_type(image):
    with torch.no_grad():  # Menonaktifkan gradient calculation
        inputs = preprocess_image(image)  # Melakukan preprocessing pada gambar
        outputs = model(inputs)  # Mendapatkan output dari model
        _, predicted = torch.max(outputs, 1)  # Mengambil prediksi dengan nilai tertinggi
        return predicted.item()  # Mengembalikan prediksi
    
# Penjelasan untuk setiap bahan
def ingredient_explanation(ingredient):
    explanations = {
        "Alcohol": "Dapat mengeringkan kulit dan menghilangkan minyak alami kulit, yang dapat memperburuk kondisi kulit kering.",
        "Fragrance": "Dapat menyebabkan iritasi, terutama bagi kulit sensitif atau kering.",
        "Benzoyl Peroxide": "Efektif untuk mengobati jerawat tetapi bisa mengeringkan kulit, terutama bagi mereka dengan kulit kering atau sensitif.",
        "Mineral Oil": "Dapat menyumbat pori-pori, sehingga tidak cocok untuk kulit berminyak atau yang rentan berjerawat.",
        "Lanolin": "Merupakan emolien yang dapat menjebak minyak, sehingga kurang cocok untuk kulit berminyak.",
        "Petrolatum": "Meskipun mengunci kelembapan, dapat terlalu berat untuk kulit berminyak dan mungkin menyumbat pori-pori.",
        "Hyaluronic Acid": "Humektan yang membantu kulit mempertahankan kelembapan, ideal untuk kulit kering.",
        "Ceramides": "Membantu memperbaiki penghalang kulit, mempertahankan kelembapan, dan mencegah kekeringan.",
        "Squalane": "Minyak ringan yang melembabkan kulit tanpa membuatnya berminyak, cocok untuk kulit kering dan normal.",
        "Salicylic Acid": "Asam beta-hidroksi (BHA) yang mengelupas dan membersihkan pori-pori, ideal untuk kulit berminyak atau rentan jerawat.",
        "Niacinamide": "Bahan anti-inflamasi yang mengatur produksi minyak dan memperbaiki fungsi penghalang kulit.",
        "Clay": "Menyerap minyak berlebih dari kulit, sangat cocok untuk kulit berminyak.",
        "Glycerin": "Bahan pelembab yang menarik kelembapan ke dalam kulit, cocok untuk semua jenis kulit.",
        "Centella Asiatica": "Menenangkan iritasi dan membantu memperbaiki penghalang kulit, bermanfaat untuk kulit sensitif atau kering.",
        "Aloe Vera": "Memiliki sifat menenangkan dan melembabkan, cocok untuk kulit normal dan sensitif.",
        "Retinol": "Turunan vitamin A yang mendorong pergantian sel dan membantu mengurangi tanda-tanda penuaan serta jerawat.",
        "Vitamin C": "Antioksidan yang mencerahkan kulit dan membantu mengurangi hiperpigmentasi.",
        "Tea Tree Oil": "Antiseptik alami yang membantu mengobati jerawat dan menenangkan peradangan.",
        "Peptides": "Asam amino yang membantu membentuk protein di kulit, mendorong kekencangan, dan mengurangi kerutan.",
        "Zinc": "Mineral yang membantu mengatur produksi minyak dan dapat menenangkan kulit berjerawat.",
        "Lactic Acid": "Asam alfa hidroksi (AHA) yang secara lembut mengelupas dan melembabkan kulit, ideal untuk kulit kering.",
        "Argan Oil": "Minyak kaya asam lemak dan vitamin E, sangat baik untuk melembabkan kulit kering.",
        "Witch Hazel": "Astringent alami yang dapat membantu mengurangi peradangan dan mengencangkan pori-pori, cocok untuk kulit berminyak.",
        "Cucumber Extract": "Dikenal karena sifat menenangkan dan melembabkan, sempurna untuk kulit sensitif.",
        "Willow Bark": "Sumber alami asam salisilat yang membantu mengatasi jerawat dan mengelupas kulit.",
        "Harsh Sulfates": "Bahan pembersih yang bisa terlalu kuat, terutama bagi kulit sensitif atau kering, dan dapat menyebabkan iritasi atau kekeringan.",
        "Panthenol": "Dikenal juga sebagai provitamin B5, bahan ini membantu menenangkan, melembabkan, dan memperbaiki penghalang kulit.",
        "Cholesterol": "Bahan pelembab yang membantu memperbaiki penghalang kulit, menjaga kelembapan dan elastisitas kulit.",
        "Azelaic Acid": "Membantu mengurangi kemerahan dan mengobati jerawat, serta dapat mencerahkan hiperpigmentasi.",
        "Licorice Root": "Ekstrak akar manis yang memiliki sifat pencerah alami dan membantu mengurangi bintik hitam.",
        "Alpha Arbutin": "Bahan yang membantu mengurangi produksi melanin dan cocok untuk tujuan pencerahan kulit.",
        "Kojic Acid": "Bahan alami yang membantu mencerahkan kulit dan mengurangi hiperpigmentasi.",
    }
    return explanations.get(ingredient, "Tidak ada penjelasan yang tersedia.")

# Fungsi untuk menyarankan bahan kimia
def suggest_chemicals(skin_type, goal):
    chemicals_to_avoid = {
        0: ["Alcohol", "Fragrance"],
        1: ["Mineral Oil", "Petrolatum"],
        2: ["Harsh Sulfates", "Fragrance"]
    }
    chemicals_to_use = {
        0: ["Hyaluronic Acid", "Ceramides"],
        1: ["Salicylic Acid", "Niacinamide"],
        2: ["Glycerin", "Centella Asiatica"]
    }
    goal_suggestions = {
        "Skin Barrier Repair": {0: ["Ceramides"], 1: ["Niacinamide"], 2: ["Panthenol"]},
        "Acne Treatment": {0: ["Azelaic Acid"], 1: ["Benzoyl Peroxide"], 2: ["Tea Tree Oil"]},
        "Skin Whitening": {0: ["Vitamin C"], 1: ["Alpha Arbutin"], 2: ["Kojic Acid"]}
    }
    return chemicals_to_avoid[skin_type], chemicals_to_use[skin_type], goal_suggestions[goal][skin_type]

# Rutinitas perawatan kulit
def skincare_routine(skin_type):
    routines = {
        0: {"Morning": ["Gentle Cleanser", "Hyaluronic Acid", "Moisturizer", "Sunscreen"],
            "Evening": ["Cream Cleanser", "Retinol", "Moisturizer"]},
        1: {"Morning": ["Foaming Cleanser", "Niacinamide", "Moisturizer", "Sunscreen"],
            "Evening": ["Gel Cleanser", "Salicylic Acid Treatment", "Oil-Free Moisturizer"]},
        2: {"Morning": ["Gentle Cleanser", "Vitamin C Serum", "Moisturizer", "Sunscreen"],
            "Evening": ["Gentle Cleanser", "Peptides Serum", "Moisturizer"]}
    }
    return routines[skin_type]

# Initialize
if 'landing_done' not in st.session_state:
    st.session_state['landing_done'] = False
if 'image_confirmed' not in st.session_state:
    st.session_state['image_confirmed'] = False
if 'show_about' not in st.session_state:
    st.session_state['show_about'] = False

# Landing Page
if not st.session_state['landing_done']:
    st.title("Selamat Datang di SkinPath 👩‍⚕️")
    st.markdown("""
        ### Fitur Utama:
        - 🧴 **Deteksi Tipe Kulit**: Unggah foto atau gunakan kamera untuk analisis tipe kulit.
        - 🔍 **Tujuan Perawatan**: Pilih tujuan perawatan untuk rekomendasi skincare yang spesifik.
        
        ### Cara Penggunaan:
        1. **Pilih Metode Input**: Di panel samping, pilih apakah Anda ingin mengambil gambar langsung dengan kamera atau mengunggah foto dari perangkat.
        2. **Unggah atau Ambil Foto**: Pastikan wajah terlihat jelas tanpa bayangan atau gangguan pencahayaan.
        3. **Konfirmasi Foto**: Setelah mengunggah atau mengambil foto, klik **Confirm Image** untuk melanjutkan ke analisis tipe kulit.
        4. **Lihat Rekomendasi**: Setelah analisis, aplikasi akan menampilkan tipe kulit dan memberikan rekomendasi produk sesuai tujuan perawatan yang Anda pilih.
        5. **Mulai Ulang**: Jika ingin mengulangi analisis, tekan tombol **Restart Analysis** di bagian bawah halaman.
        
        ### Disclaimer
        Informasi yang diberikan di sini hanya sebagai referensi. Harap dicatat bahwa terdapat variabel penentu lain, seperti sensitivitas kulit, yang mungkin tidak didukung oleh aplikasi ini. Pastikan untuk berkonsultasi dengan profesional kesehatan atau dermatologis sebelum mengambil keputusan terkait perawatan kulit.
        
        > **Tip**: Untuk hasil terbaik, gunakan foto dengan pencahayaan alami dan wajah bersih tanpa riasan berlebih.
        """)
    
    if st.button("Lanjutkan ke Analisis"):
        st.session_state['landing_done'] = True
        st.rerun()

else:
    st.title("SkinPath App 👩‍⚕️")
    st.sidebar.header("Input Method")
    input_option = st.sidebar.selectbox("Select Image Input:", ("📸Capture Image", "📄Upload Photo"))

    # Tambahkan tombol About ke sidebar
    if st.sidebar.button("ℹ️ About"):
        st.session_state['show_about'] = not st.session_state['show_about']

    # Menampilkan panel "About" jika statusnya True
    if st.session_state['show_about']:
        with st.sidebar.expander("About This App", expanded=True):
            st.write("""SkinPath adalah aplikasi analisis kulit berbasis AI yang menggunakan model ResNet50 untuk mendeteksi tipe kulit dari gambar yang diunggah pengguna.
            Aplikasi ini memberikan saran skincare yang disesuaikan dengan tipe kulit dan tujuan perawatan pengguna, sehingga membantu memilih produk skincare
            yang sesuai dengan kebutuhan kulit.
            """)
            st.markdown("**Author**: Rheisan Firnandatama")
            st.markdown("**NIM**: 22537141021")

    # Menampilkan instruksi kamera untuk panduan pengguna
    if input_option == "📸Capture Image":
        st.info("Pastikan wajah Anda terlihat jelas tanpa bayangan yang menghalangi, dan pencahayaan yang cukup agar hasil lebih maksimal.")

    image = st.camera_input("Take a photo") if input_option == "📸Capture Image" else st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

    # Menampilkan pratinjau gambar dan meminta konfirmasi
    if image and not st.session_state['image_confirmed']:
        st.markdown("### Image Preview:")
        st.image(image, caption="Your Image", use_column_width=True)
        if st.button("Confirm Image"):
            st.session_state['image_confirmed'] = True
            img_bytes = image.read() if hasattr(image, 'read') else image.getvalue()
            skin_type = predict_skin_type(img_bytes)
            st.session_state['skin_type'] = skin_type
            st.rerun()

    # Lanjutkan analisis hanya setelah konfirmasi
    elif st.session_state['image_confirmed']:
        skin_types = ["Dry", "Oily", "Normal"]
        st.success(f"Detected Skin Type: **{skin_types[st.session_state['skin_type']]}**")
        
        routine = skincare_routine(st.session_state['skin_type'])
        st.markdown("### Skincare Routine Suggestions 🧴")
        morning, evening = st.columns(2)

        with morning:
            st.subheader("🌞 Morning Routine")
            for step in routine["Morning"]:
                st.write(f"🔹 {step}")

        with evening:
            st.subheader("🌜 Evening Routine")
            for step in routine["Evening"]:
                st.write(f"🔹 {step}")

        # Pemilihan Tujuan
        st.header("Set Your Skincare Goal 🎯")
        goal = st.selectbox("Choose your skincare goal:", ["Skin Barrier Repair", "Acne Treatment", "Skin Whitening"])

        # Saran Bahan Kimia
        avoid, recommend, additional = suggest_chemicals(st.session_state['skin_type'], goal)
        st.markdown("### Ingredients to Avoid 🚫")
        st.write(", ".join(avoid))

        st.markdown("### Recommended Ingredients ✅")
        st.write(", ".join(recommend))

        st.markdown("### Additional Ingredients for Goal 💡")
        st.write(", ".join(additional))

        # Menggabungkan bahan tanpa duplikasi
        unique_ingredients = list(set(avoid + recommend + additional))

        # Expanders untuk Penjelasan Bahan tanpa duplikasi
        st.markdown("### Ingredient Explanations 📖")
        for ingredient in unique_ingredients:
            with st.expander(f"{ingredient} 📜"):
                st.write(ingredient_explanation(ingredient))
        
        if st.button("Restart Analysis"):
            st.session_state.clear()
            st.rerun()