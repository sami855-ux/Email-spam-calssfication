# Building a Spam Classifier with FastAPI

Deployed Website Link: https://email-spam-calssfication-5.onrender.com

![Screenshot 2025-02-09 101057](https://github.com/user-attachments/assets/75043eb2-5aad-477c-8d64-058faccda828)

## How To run this project

step 1: install all the dependencies from requirements.txt using thic command
```python
    pip install -r requirements.txt
```
step 2: Run the train python file in src folder
```python
    python src/train.py
```
step 3: Run the predict python file in src folder
```python
    python src/predict.py
```
## File Structure
    .  
├── api                  # main file found here (main.py)
├── data                 # csv file for the model  
├── model                # the outputed model file in joblib formate  
├── notebooks            # Data exploration file for visualization 
├── src                  # Train and predict files 
├── static               # CSS files 
├── templates            # HTML files 
├── .gitignore           # for git to ignore some files 
├── README.md            
├── Report.pdf           # Report documentation is added  
└── requirements.txt     # All the dependencies in this project


## Try this message

### For spam
Dear Winner,  We are pleased to inform you that you have won a cash prize of $1,000,000 in the XYZ Lottery!To claim your prize, please provide your bank details and contact information to process your winnings.  Best regards,  Lottery Coordinator

### For not spam

Hello,We received a request to reset your password. If you made this request, please click the link below to reset your password:  If you did not request a password reset, you can ignore this message.  Thank you,  The Support Team 

