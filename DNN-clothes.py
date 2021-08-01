import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import linear_model

#print(training_data)
#print(training_labels)

def acu_curve(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.show()

if __name__ == '__main__':
    hospital_lab = pd.read_pickle('./data/hospital_lab.pkl')
    ## DNN
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(43,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2,activation='Softmax')
    ])
    data_train , data_test , labels_train, labels_test=train_test_split(data,labels,test_size=0.3,random_state=0)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(data_train, labels_train, epochs=10);
    #test_loss, test_acc=model.evaluate(data_test,  labels_test, verbose=2)
    #print('\nTest accuracy:', test_acc)
    probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    #predictions=probability_model.predict(data_test)
    #acu_curve(labels_test,predictions[:,1])

    # Liner regression
    for name,grouped in patienthealthsystemstayid_time_lab.groupby('patienthealthsystemstayid'):
        linear_x=grouped.dis_lastday.values
        linear_y=probability_model.predict(grouped[['AST (SGOT)', 'alkaline phos.', 'total bilirubin', 'anion gap', 'BUN', 'calcium', 'glucose', 'sodium', 'total protein', 'ALT (SGPT)', 'albumin', 'bicarbonate', 'chloride', 'potassium', '-eos', '-basos', '-lymphs', '-polys', '-monos', 'bedside glucose', 'PT', 'PT - INR', 'lactate', 'FiO2', 'Base Excess', 'pH', 'paO2', 'paCO2', 'HCO3', 'magnesium', 'O2 Sat (%)', 'phosphate', 'MPV', 'creatinine', 'MCHC', 'platelets x 1000', 'RDW', 'WBC x 1000', 'MCH', 'RBC', 'MCV', 'Hgb', 'Hct']].values)[:,1]
        #print(linear_y)
        linear_x = np.array(linear_x).reshape(-1, 1)
        linear_y = np.array(linear_y).reshape(-1, 1)
        clf = linear_model.LinearRegression();
        clf.fit(linear_x, linear_y);
        predict_data_y = clf.predict(np.array(0).reshape(-1,1))
        index=df.shape[0]
        df.loc[index,'patienthealthsystemstayid']=name;
        df.loc[index,'pre_lables']=predict_data_y[0][0];



