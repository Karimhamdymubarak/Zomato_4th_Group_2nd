
import streamlit as st
import pandas as pd
import joblib
import sklearn
import category_encoders

Model = joblib.load("Group_4_Model.pkl")
Inputs = joblib.load("Group_4_Inputs.pkl")

def Make_Prediction(online_order,book_table,location,approx_cost,listed_in_type,listed_in_city,rest_type_counts,cuisines_counts):
    df_pred = pd.DataFrame(columns=Inputs)
    df_pred.at[0,"online_order"] = online_order
    df_pred.at[0,"book_table"] = book_table
    df_pred.at[0,"location"] = location
    df_pred.at[0,"approx_cost(for two people)"] = approx_cost
    df_pred.at[0,"listed_in(type)"] = listed_in_type
    df_pred.at[0,"listed_in(city)"] = listed_in_city
    df_pred.at[0,"rest_type_counts"] = rest_type_counts
    df_pred.at[0,"cuisines_counts"] = cuisines_counts
    result = Model.predict(df_pred)
    return result[0]

def main():
    st.title("Zomato Resturants")
    online_order = st.selectbox("online_order" , ['Yes', 'No'])
    book_table = st.selectbox("book_table" , ['Yes', 'No'])
    location = st.selectbox("location" ,['Banashankari', 'Basavanagudi', 'other', 'Jayanagar', 'JP Nagar',
       'Bannerghatta Road', 'BTM', 'Electronic City', 'Shanti Nagar',
       'Koramangala 5th Block', 'Richmond Road', 'HSR',
       'Koramangala 7th Block', 'Bellandur', 'Sarjapur Road',
       'Marathahalli', 'Whitefield', 'Old Airport Road', 'Indiranagar',
       'Koramangala 1st Block', 'Frazer Town', 'MG Road', 'Brigade Road',
       'Lavelle Road', 'Church Street', 'Ulsoor', 'Residency Road',
       'Shivajinagar', 'St. Marks Road', 'Cunningham Road',
       'Commercial Street', 'Vasanth Nagar', 'Domlur',
       'Koramangala 8th Block', 'Ejipura', 'Jeevan Bhima Nagar',
       'Kammanahalli', 'Koramangala 6th Block', 'Brookefield',
       'Koramangala 4th Block', 'Banaswadi', 'Kalyan Nagar',
       'Malleshwaram', 'Rajajinagar', 'New BEL Road'])
    approx_cost = st.slider("approx_cost" , min_value=10 , max_value =10000 , value = 800, step = 200)
    listed_in_type = st.selectbox("listed_in_type" , ['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out',
       'Drinks & nightlife', 'Pubs and bars'])
    listed_in_city = st.selectbox("listed_in_city" , ['Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
       'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
       'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
       'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
       'Koramangala 4th Block', 'Koramangala 5th Block',
       'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
       'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',
       'Old Airport Road', 'Rajajinagar', 'Residency Road',
       'Sarjapur Road', 'Whitefield'])
    rest_type_counts = st.selectbox("rest_type_counts" ,[1, 2] )
    cuisines_counts = st.selectbox("cuisines_counts" , [3, 2, 1, 4, 5, 8, 7, 6])
    if st.button("Predict"):
        results= Make_Prediction(online_order,book_table,location,approx_cost,listed_in_type,listed_in_city,rest_type_counts,cuisines_counts)
        List_Types = ["Your Resturant may fail" , "Your Resturant will success"]
        st.text(List_Types[results])
main()



    
    
