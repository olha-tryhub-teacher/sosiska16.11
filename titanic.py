import pandas as pd  # Імпортуємо бібліотеку Pandas для роботи з даними
from sklearn.model_selection import train_test_split  # Імпортуємо функцію для розділення даних на тренувальну і тестову вибірки
from sklearn.preprocessing import StandardScaler  # Імпортуємо StandardScaler для нормалізації даних
from sklearn.neighbors import KNeighborsClassifier  # Імпортуємо K-Nearest Neighbors Classifier
from sklearn.metrics import confusion_matrix, accuracy_score  # Імпортуємо метрики оцінки моделі

#  pip install scikit-learn
# Крок 1. Завантаження та очищення даних
df = pd.read_csv("titanic.csv")  # Завантажуємо дані з файлу Titanic у DataFrame
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)  # Видаляємо неважливі для моделі стовпці

# Кодуємо стовпець "Embarked" за допомогою one-hot encoding
df[list(pd.get_dummies(df["Embarked"]).columns)] = pd.get_dummies(df["Embarked"])
df.fillna({"Embarked": "S"}, inplace=True)  # Заповнюємо пропущені значення в "Embarked" значенням "S"
df.drop("Embarked", axis=1, inplace=True)  # Видаляємо оригінальний стовпець "Embarked"

# Розраховуємо медіанний вік для кожного класу пасажирів
age_1 = df[df["Pclass"] == 1]["Age"].median()  # Медіанний вік пасажирів 1-го класу
age_2 = df[df["Pclass"] == 2]["Age"].median()  # Медіанний вік пасажирів 2-го класу
age_3 = df[df["Pclass"] == 3]["Age"].median()  # Медіанний вік пасажирів 3-го класу

print(age_1, age_2, age_3)

# Функція для заповнення пропущених значень віку відповідно до класу пасажира
def fill_age(row):
    if pd.isnull(row["Age"]):  # Якщо значення віку пропущено
        if row["Pclass"] == 1:
            return age_1  # Використовуємо медіанний вік 1-го класу
        if row["Pclass"] == 2:
            return age_2  # Використовуємо медіанний вік 2-го класу
        return age_3  # Використовуємо медіанний вік 3-го класу
    return row["Age"]  # Повертаємо наявне значення віку

df["Age"] = df.apply(fill_age, axis=1)  # Заповнюємо пропущені значення віку у всіх рядках

# Функція для перетворення статі на числовий формат
def fill_Gender(Gender):
    if Gender == "male":
        return 1  # Чоловік - 1
    return 0  # Жінка - 0

df["Gender"] = df["Gender"].apply(fill_Gender)  # Застосовуємо функцію до стовпця "Gender"

X = df.drop("Survived", axis = 1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classfilter = KNeighborsClassifier(n_neighbors=10)
classfilter.fit(X_train, y_train)

y_pred = classfilter.predict(X_test)
print("Відсоток правильно передбачених результатів:", accuracy_score(y_test, y_pred) * 100)
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
