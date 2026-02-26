import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay


Border = "="*40

########################################################################
# STEP 1: Load dataset and display basic information
########################################################################

print(Border)
print("STEP 1: Load dataset and display basic information")
print(Border)

Dataset = "student_performance_ml.csv"
df = pd.read_csv(Dataset)

print("\nFirst 5 Records:")
print(df.head())

print("\nLast 5 Records:")
print(df.tail())

print("\nTotal Rows and Columns:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


########################################################################
# STEP 2: Total students, Passed and Failed count
########################################################################

print(Border)
print("STEP 2: Total students, Passed and Failed count")
print(Border)

total_students = len(df)
passed_students = df[df['FinalResult'] == 1].shape[0]
failed_students = df[df['FinalResult'] == 0].shape[0]

print("Total Students:", total_students)
print("Passed Students:", passed_students)
print("Failed Students:", failed_students)


########################################################################
# STEP 3: Calculate averages and statistics
########################################################################

print(Border)
print("STEP 3: Calculate averages and statistics")
print(Border)

print("Average Study Hours:", df['StudyHours'].mean())
print("Average Attendance:", df['Attendance'].mean())
print("Maximum Previous Score:", df['PreviousScore'].max())
print("Minimum Sleep Hours:", df['SleepHours'].min())


########################################################################
# STEP 4: Distribution and percentage of Pass/Fail
########################################################################

print(Border)
print("STEP 4: Distribution and percentage of Pass/Fail")
print(Border)

result_count = df['FinalResult'].value_counts()
percentage = df['FinalResult'].value_counts(normalize=True) * 100

print("Result Count:")
print(result_count)

print("\nPercentage:")
print(percentage)

if abs(percentage[1] - percentage[0]) < 10:
    print("\nDataset is Balanced")
else:
    print("\nDataset is NOT Balanced")


########################################################################
# STEP 5: Observations
########################################################################

print(Border)
print("STEP 5: Observations")
print(Border)

study_pass_avg = df[df['FinalResult'] == 1]['StudyHours'].mean()
study_fail_avg = df[df['FinalResult'] == 0]['StudyHours'].mean()

attendance_pass_avg = df[df['FinalResult'] == 1]['Attendance'].mean()
attendance_fail_avg = df[df['FinalResult'] == 0]['Attendance'].mean()

print("Average StudyHours (Pass):", study_pass_avg)
print("Average StudyHours (Fail):", study_fail_avg)

print("Average Attendance (Pass):", attendance_pass_avg)
print("Average Attendance (Fail):", attendance_fail_avg)

if study_pass_avg > study_fail_avg:
    print("Conclusion: Higher StudyHours increase passing chances")

if attendance_pass_avg > attendance_fail_avg:
    print("Conclusion: Higher Attendance improves FinalResult")


########################################################################
# STEP 6: Visualization
########################################################################

print(Border)
print("STEP 6: Visualization")
print(Border)

plt.figure()
plt.hist(df['StudyHours'])
plt.title("Histogram of Study Hours")
plt.xlabel("Study Hours")
plt.ylabel("Students")
plt.show()

plt.figure()
plt.scatter(df['StudyHours'], df['PreviousScore'])
plt.title("StudyHours vs PreviousScore")
plt.xlabel("StudyHours")
plt.ylabel("PreviousScore")
plt.show()


########################################################################
# STEP 7: Prepare Features and Target
########################################################################

print(Border)
print("STEP 7: Prepare Features and Target")
print(Border)

X = df[['StudyHours', 'Attendance', 'PreviousScore',
        'AssignmentsCompleted', 'SleepHours']]

y = df['FinalResult']


########################################################################
# STEP 8: Train-Test Split
########################################################################

print(Border)
print("STEP 8: Train-Test Split")
print(Border)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


########################################################################
# STEP 9: Train Decision Tree Model
########################################################################

print(Border)
print("STEP 9: Model Training")
print(Border)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)

print("Model trained successfully")


########################################################################
# STEP 10: Prediction
########################################################################

print(Border)
print("STEP 10: Prediction")
print(Border)

Y_pred = model.predict(X_test)

result_df = pd.DataFrame({
    "Actual": Y_test.values,
    "Predicted": Y_pred
})

print(result_df)


########################################################################
# STEP 11: Accuracy Calculation
########################################################################
print(Border)
print("STEP 11: Accuracy Calculation")
print(Border)

accuracy = accuracy_score(Y_test, Y_pred)
print("Testing Accuracy = {:.2f}%".format(accuracy * 100))


########################################################################
# STEP 12: Confusion Matrix
########################################################################

print(Border)
print("STEP 12: Confusion Matrix")
print(Border)

cm = confusion_matrix(Y_test, Y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


########################################################################
# STEP 13: Predict New Student Result
########################################################################

print(Border)
print("STEP 13: Predict New Student Result")
print(Border)

new_student = [[6, 85, 66, 7, 7]]

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Result: PASS")
else:
    print("Result: FAIL")


########################################################################
# FINAL CONCLUSION
########################################################################

print(Border)
print("FINAL CONCLUSION")
print(Border)

print("""
Decision Tree model successfully trained.
Model predicts student performance using:
• StudyHours
• Attendance
• PreviousScore
• AssignmentsCompleted
• SleepHours
Higher StudyHours and Attendance improve passing probability.
Machine Learning helps predict student results accurately.
""")

print(Border)
print("PROGRAM COMPLETED SUCCESSFULLY")
print(Border)