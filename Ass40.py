import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix


Border = "=" *40


print(Border)
print("ADVANCED DECISION TREE - STUDENT PERFORMANCE PREDICTION")
print("MCA Machine Learning Practical")
print("Name: Aditi Shinde")
print(Border)


########################################################################
# STEP 1: Load Dataset
########################################################################

print(Border)
print("STEP 1: Load Dataset")
print(Border)

df = pd.read_csv("student_performance_ml.csv")

print("\nDataset Loaded Successfully")
print(df.head())


features = ['StudyHours', 'Attendance', 'PreviousScore',
            'AssignmentsCompleted', 'SleepHours']

X = df[features]
y = df['FinalResult']


X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)

print("\nOriginal Model Accuracy:", accuracy * 100)


########################################################################
# STEP 2: Feature Importance
########################################################################

print(Border)
print("STEP 2: Feature Importance")
print(Border)

importance = model.feature_importances_

for i in range(len(features)):
    print(features[i], ":", importance[i])

max_index = importance.argmax()
min_index = importance.argmin()

print("\nMost Important Feature:", features[max_index])
print("Least Important Feature:", features[min_index])


########################################################################
# STEP 3: Remove SleepHours and train again
########################################################################

print(Border)
print("STEP 3: Removing SleepHours and Retraining")
print(Border)

features2 = ['StudyHours', 'Attendance', 'PreviousScore',
             'AssignmentsCompleted']

X2 = df[features2]

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(
    X2, y, test_size=0.2, random_state=42)

model2 = DecisionTreeClassifier(random_state=42)
model2.fit(X_train2, Y_train2)

Y_pred2 = model2.predict(X_test2)

accuracy2 = accuracy_score(Y_test2, Y_pred2)

print("Accuracy without SleepHours:", accuracy2 * 100)


########################################################################
# STEP 4: Train using only StudyHours and Attendance
########################################################################

print(Border)
print("STEP 4: Using only StudyHours and Attendance")
print(Border)

features3 = ['StudyHours', 'Attendance']

X3 = df[features3]

X_train3, X_test3, Y_train3, Y_test3 = train_test_split(
    X3, y, test_size=0.2, random_state=42)

model3 = DecisionTreeClassifier(random_state=42)
model3.fit(X_train3, Y_train3)

accuracy3 = model3.score(X_test3, Y_test3)

print("Accuracy with only 2 features:", accuracy3 * 100)


########################################################################
# STEP 5: Predict New Students
########################################################################

print(Border)
print("STEP 5: Predict New Students")
print(Border)

new_students = pd.DataFrame({
    'StudyHours': [2, 5, 7, 4, 8],
    'Attendance': [60, 75, 90, 70, 95],
    'PreviousScore': [45, 60, 80, 55, 85],
    'AssignmentsCompleted': [3, 6, 9, 5, 10],
    'SleepHours': [5, 6, 8, 6, 7]
})

predictions = model.predict(new_students)

new_students['Prediction'] = predictions

print(new_students)

print("\n0 = Fail, 1 = Pass")


########################################################################
# STEP 6: Manual Accuracy Calculation
########################################################################

print(Border)
print("STEP 6: Manual Accuracy Calculation")
print(Border)

correct = 0

for i in range(len(Y_test)):
    if Y_test.iloc[i] == y_pred[i]:
        correct += 1

manual_accuracy = correct / len(Y_test)

print("Manual Accuracy:", manual_accuracy * 100)
print("Sklearn Accuracy:", accuracy * 100)


########################################################################
# STEP 7: Misclassified Students
########################################################################

print(Border)
print("STEP 7: Misclassified Students")
print(Border)

misclassified = X_test[Y_test != y_pred]

print(misclassified)

print("Number of Misclassified:", len(misclassified))


########################################################################
# STEP 8: Random State Comparison
########################################################################

print(Border)
print("STEP 8: Random State Comparison")
print(Border)

states = [0, 10, 42]

for s in states:

    X_train_s, X_test_s, Y_train_s, Y_test_s = train_test_split(
        X, y, test_size=0.2, random_state=s)

    model_s = DecisionTreeClassifier(random_state=s)
    model_s.fit(X_train_s, Y_train_s)

    acc = model_s.score(X_test_s, Y_test_s)

    print("Random State", s, "Accuracy:", acc * 100)


########################################################################
# STEP 9: Decision Tree Visualization
########################################################################

print(Border)
print("STEP 9: Decision Tree Visualization")
print(Border)

plt.figure(figsize=(12, 8))

plot_tree(model,
          feature_names=features,
          class_names=['Fail', 'Pass'],
          filled=True)

plt.show()


########################################################################
# STEP 10: Add PerformanceIndex Feature
########################################################################

print(Border)
print("STEP 10: Adding PerformanceIndex Feature")
print(Border)

df['PerformanceIndex'] = (df['StudyHours'] * 2) + df['Attendance']

features4 = ['StudyHours', 'Attendance', 'PreviousScore',
             'AssignmentsCompleted', 'SleepHours', 'PerformanceIndex']

X4 = df[features4]

X_train4, X_test4, Y_train4, Y_test4 = train_test_split(
    X4, y, test_size=0.2, random_state=42)

model4 = DecisionTreeClassifier(random_state=42)
model4.fit(X_train4, Y_train4)

accuracy4 = model4.score(X_test4, Y_test4)

print("Accuracy with PerformanceIndex:", accuracy4 * 100)


########################################################################
# STEP 11: Overfitting Check (max_depth=None)
########################################################################

print(Border)
print("STEP 11: Overfitting Check")
print(Border)

model5 = DecisionTreeClassifier(max_depth=None, random_state=42)

model5.fit(X_train, Y_train)

train_acc = model5.score(X_train, Y_train)
test_acc = model5.score(X_test, Y_test)

print("Training Accuracy:", train_acc * 100)
print("Testing Accuracy:", test_acc * 100)

if train_acc == 1.0 and test_acc < 1.0:
    print("Model is Overfitting")
else:
    print("Model is Balanced")


########################################################################
# END OF PROGRAM
########################################################################

print(Border)
print("PROGRAM COMPLETED SUCCESSFULLY")
print(Border)