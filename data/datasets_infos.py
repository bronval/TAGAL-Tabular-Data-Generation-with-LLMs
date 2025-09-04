
datasets_infos = {

    "adult": {
        "url": "",
        "target": "income",
        "task": "classification",
        "pos_label": ">50K",
        "categorical_features": ["workclass", "education", "marital.status", "occupation", "relationship", "race", "sex", "native.country"],
        "drop_features" : [],
        "rename_features" : {},
        "filename": "adult.csv",
    },

    "bank": {
        "url": "",
        "target": "y",
        "task": "classification",
        "pos_label": "yes",
        "categorical_features": ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"],
        "drop_features" : [],
        "rename_features" : {},
        "filename": "bank.csv",
    },

    "thyroid": {
        "url": "",
        "target": "Recurred",
        "task": "classification",
        "pos_label": "Yes",
        "categorical_features": ["Gender","Smoking","Hx Smoking","Hx Radiothreapy","Thyroid Function","Physical Examination","Adenopathy","Pathology","Focality","Risk","T","N","M","Stage","Response","Recurred"],
        "drop_features" : [],
        "rename_features" : {},
        "filename": "thyroid.csv",
    },

    "german": {
        "url": "",
        "target": "Risk",
        "task": "classification",
        "pos_label": "good",
        "categorical_features": ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"],
        "drop_features" : [],
        "rename_features" : {},
        "filename": "german.csv",
    },

    "sick": {
        "url": "",
        "target": "Class",
        "task": "classification",
        "pos_label": "sick",
        "categorical_features": ["sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "sick", "pregnant",
                                 "thyroid_surgery", "I131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium",
                                 "goitre", "tumor", "hypopituitary", "psych", "TSH_measured", "T3_measured", "TT4_measured",
                                 "T4U_measured", "FTI_measured", "referral_source"],
        "drop_features" : [],
        "rename_features" : {},
        "filename": "sick.csv",
    },

    "travel": {
        "url": "https://www.kaggle.com/datasets/tejashvi14/tour-travels-customer-churn-prediction",
        "target": "ChurnOrNot",
        "task": "classification",
        "pos_label": "1",
        "categorical_features": ["FrequentFlyer", "AnnualIncomeClass", "AccountSyncedToSocialMedia", "BookedHotelOrNot"],
        "drop_features" : [],
        "rename_features" : {},
        "filename": "travel.csv",
    }

}

