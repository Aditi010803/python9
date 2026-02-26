
import pandas as pd
import matplotlib.pyplot as plt

Border = "-"*40
########################################################################
# 1. Load dataset and display basic information
########################################################################
print(Border)
print("Load dataset and display basic information")
print(Border)

Dataset ="student_performance_ml.csv"
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
# 2. Total students,Passed and Failed count
########################################################################

print(Border)
print("Total students, Passed and Failed count")
print(Border)


total_students = len(df)
print("Total Students:", total_students)

passed_students = df[df['FinalResult'] == 1].shape[0]
print("Passed Students:", passed_students)

failed_students = df[df['FinalResult'] == 0].shape[0]
print("Failed Students:", failed_students)

########################################################################
# 3. Calculate averages and statistics
########################################################################

print(Border)
print("Calculate averages and statistics")
print(Border)

avg_study = df['StudyHours'].mean()
avg_attendance = df['Attendance'].mean()
max_previous = df['PreviousScore'].max()
min_sleep = df['SleepHours'].min()

print("Average Study Hours:", avg_study)
print("Average Attendance:", avg_attendance)
print("Maximum Previous Score:", max_previous)
print("Minimum Sleep Hours:", min_sleep)


########################################################################
# 4. Distribution and percentage of Pass/Fail
########################################################################

print(Border)
print("Distribution and percentage of Pass/Fail")
print(Border)

result_count = df['FinalResult'].value_counts()
print("Result Count:")
print(result_count)

percentage = df['FinalResult'].value_counts(normalize=True) * 100

print("\nPercentage:")
print(percentage)

if abs(percentage[1] - percentage[0]) < 10:
    print("\nDataset is Balanced")
else:
    print("\nDataset is NOT Balanced")


########################################################################
# 5. Observations
########################################################################

print(Border)
print("Observations")
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
    print("Conclusion: Higher StudyHours increase chance of passing")

if attendance_pass_avg > attendance_fail_avg:
    print("Conclusion: Higher Attendance improves FinalResult")


########################################################################
# 6. Histogram of StudyHours
########################################################################


plt.figure()
plt.hist(df['StudyHours'], bins=10)
plt.title("Histogram of Study Hours")
plt.xlabel("Study Hours")
plt.ylabel("Number of Students")
plt.show()


########################################################################
# 7. Scatter Plot StudyHours vs PreviousScore
########################################################################

plt.figure()

pass_students = df[df['FinalResult'] == 1]
fail_students = df[df['FinalResult'] == 0]

plt.scatter(pass_students['StudyHours'],
            pass_students['PreviousScore'],
            label="Pass")

plt.scatter(fail_students['StudyHours'],
            fail_students['PreviousScore'],
            label="Fail")

plt.title("StudyHours vs PreviousScore")
plt.xlabel("StudyHours")
plt.ylabel("PreviousScore")
plt.legend()
plt.show()


########################################################################
# 8. Boxplot for Attendance
########################################################################


plt.figure()
plt.boxplot(df['Attendance'])
plt.title("Boxplot of Attendance")
plt.ylabel("Attendance")
plt.show()

########################################################################
# 9. AssignmentsCompleted vs FinalResult
########################################################################

plt.figure()
plt.scatter(df['AssignmentsCompleted'],
            df['FinalResult'])

plt.title("AssignmentsCompleted vs FinalResult")
plt.xlabel("AssignmentsCompleted")
plt.ylabel("FinalResult")
plt.show()

########################################################################
# 10. SleepHours vs FinalResult
########################################################################


plt.figure()
plt.scatter(df['SleepHours'],
            df['FinalResult'])

plt.title("SleepHours vs FinalResult")
plt.xlabel("SleepHours")
plt.ylabel("FinalResult")
plt.show()


#######################################################################
# END OF PROGRAM
########################################################################
print(Border)
print("PROGRAM COMPLETED SUCCESSFULLY")
print(Border)
