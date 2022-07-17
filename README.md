# Neural_Network_Charity_Analysis

This project is about an application of Neurel Network Model to a Charity Analysis.

## Overview:

By the knowledge of machine learning and neural networks, I willl use the features in the provided dataset to to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup. The dataset is ![charity_data](resources/charity_data.csv)

The CSV file contains more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special consideration for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

The purpose of the work is to create a Neurel Network Model to predict the success of each record acccording to the geven features in the dataset and to optimize the model.

## Results:

### Data Preprocessing

**1.** The variable which is considered as target of the model is success, given as 'IS_SUCCESSFUL' column.

**2.** Numerical features are 'ASK_AMT' and 'STATUS'. Categorical features are 'APPLICATION_TYPE','AFFILIATION','CLASSIFICATION','USE_CASE','ORGANIZATION', 'INCOME_AMT'and 'SPECIAL_CONSIDERATIONS'.
The categorical features 'APPLICATION_TYPE' and 'CLASSIFICATION' have more than 10 values in the original data set so I reduced the number of values by using the following codes: 

For 'APPLICATION_TYPE'
>
>APPLICATION_TYPE_counts=application_df.APPLICATION_TYPE.value_counts()
>
> replace_types = list(APPLICATION_TYPE_counts[APPLICATION_TYPE_counts < 500].index)
>
> for typ in replace_types:
>   application_df.APPLICATION_TYPE = application_df.APPLICATION_TYPE.replace(typ,"Other")
>
I decideded the boundary of the rare values to put others by using the plot.density() method, the graph is as follows.

![](resources/density_application_type.jpg)

For 'CLASSIFICATION'
>
> CLASSIFICATION_counts=application_df.CLASSIFICATION.value_counts()
> 
> replace_classes = list(CLASSIFICATION_counts[CLASSIFICATION_counts < 1800].index)
>
> for clas in replace_classes:
>    application_df.CLASSIFICATION = application_df.CLASSIFICATION.replace(clas,"Other")
>
I decideded the boundary of the rare values to put others by using the density graph which is the following.

![](resources/density_classificarion.jpg)

After the number of values raduced , the categorical features are encoded by ***OneHotEncoder*** as follows.

> enc = OneHotEncoder(sparse=False)
>
> encode_df = pd.DataFrame(enc.fit_transform(application_df[application_cat]))
>
>encode_df.columns = enc.get_feature_names(application_cat)
>

An then I merged one-hot encoded features and drop the originals by using the following codes

> application_df = application_df.merge(encode_df,left_index=True, right_index=True)
>
> application_df = application_df.drop(application_cat,axis=1)
>
**3.** The columns 'EIN' and "NAME' are dropped, because they do not have any effect on the success; so they are neither feature nor target. The code is the following:

> application_df=application_df1.drop(['EIN','NAME'],axis=1)


