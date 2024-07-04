import streamlit as st
import requests

st.title('Estimateur de Prix de Maison')
st.write('Entrez les caractéristiques de la maison pour estimer son prix.')

surface_reelle_bati = st.number_input('Surface réelle bâtie (m²)')
nombre_pieces_principales = st.number_input('Nombre de pièces principales', step=1)
code_postal = st.number_input('Code postal', step=1)
type_local = st.selectbox('Type de local', ['Maison', 'Appartement'])
surface_terrain = st.number_input('Surface du terrain (m²)')

if st.button('Estimer le Prix'):
    response = requests.post("http://127.0.0.1:8000/predict", json={
        "surface_reelle_bati": surface_reelle_bati,
        "nombre_pieces_principales": nombre_pieces_principales,
        "code_postal": code_postal,
        "type_local": type_local,
        "surface_terrain": surface_terrain
    })
    result = response.json()
    st.write(f"Prix estimé : {result['estimated_price']} €")
