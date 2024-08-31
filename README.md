# Darvishism
A model based approach to analyzing the effects of adding or subtracting pitches from a pitcher's arsenal.

# Methodology:  
Assess the effectiveness of the pitcher's current arsenal  
- Developed a model to measure "deception": the pitcher's ability to miss the barrel of the bat
- Ordinal Gradient boosting model (found in stuff_model.py) with dependent variable with three classes (whiff, weak contact, barrel + sweet spot in that order)
- Deception scores were put in a matrix showing a score for each pitch and its effectiveness off of another pitch in the repertoire

Project what a new pitch shape might look like for that pitcher
- Similarity scores generated for each pitcher, based upon pitch usage, pitch shapes, and arm angles*
- XGBoost Regression trained on the 60 most similar pitchers that throw the target pitch to project the velocity, x and y movement of the target pitch
*arm angles were calculated using x and z release positions along with DBSCAN to fine-tune the calculations (DBSCAN algorithm can be found in data manipulation.R)

Analyze the effects of adding or subtracting pitches from an arsenal  
- Took the marginal effects of adding another pitch to the repertoire using OLS

Observe the overall impact of the repertoire changes  
- Used the deception matrix to calculate an arsenal grade
- Observed the change in the arsenal grade after pitch addition/subtraction

# Data used:  
Training data consisted of Statcast pitch by pitch from 2020-2022; 2023 was used for validation

# Main Findings:  
Our pitcher deception model revealed that pitches that were closer in velocity to that pitcher's primary fastball resulted in a higher probability for the contact classes. It also showed that faster pitches generally resulted in higher probabilities of the swing and miss class occurring. We were happy with the results of this model, as it not only found the value in pitches that could generate swings and misses, but in pitches (primarily fastball variants such as cutters and sinkers) that could generate weak contact as well.  

The pitch projection models were validated on pitchers from the 2023 season who added a pitch to their repertoire. What we found was that the models were mostly able to accurately predict the new pitch shapes, giving us confidence in our ability to project new pitch shapes and how they would play off existing pitches moving forward.

The marginal effects of adding another pitch showed a decrease in value, but a minimal one that could be offset with the addition of a quality pitch.

# App Link:
https://jonahsoos24.shinyapps.io/Darvishism_Pitcher_Arsenal_Evaluation_App/
