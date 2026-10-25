import pickle


#New data for prediction
customer = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}


#Loading the model
with open('pipeline.bin', 'rb') as f_in:
    model = pickle.load(f_in)

#Prediction
pred = model.predict_proba([customer])[:,1].item()

print(f'Conversion_probability: {pred:.3f}')
