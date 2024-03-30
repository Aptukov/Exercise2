# Импортируем нужные библиотеки
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Загрузка данных
train_data = pd.read_csv('train.csv', sep=',')
test_data = pd.read_csv('test.csv')

# Убираем NaN значения
train_data['text'] = train_data['text'].fillna('')
test_data['text'] = test_data['text'].fillna('')

# Создаём функцию для удаления круглых скобок - {}
def remove_braces(string):
   trans_table = {ord('{'): None, ord('}'): None}
   return string.translate(trans_table)

# Применяем функцию remove_braces ко всему столбцу 'labels'
train_data['labels'] = train_data['labels'].apply(remove_braces)

# Преобразование текста в числовые признаки с использованием TF-IDF векторизации
vectorizer = TfidfVectorizer(max_features=200)
X_train_tfidf = vectorizer.fit_transform(train_data['text'])

# Подготовка меток
labels = train_data['labels'].str.get_dummies(sep=', ')

# Разделение данных на обучающую и валидационную выборку
X_train, X_val, y_train, y_val = train_test_split(X_train_tfidf, labels, test_size=0.2, random_state=42)

# Обучение модели RandomForestClassifier для мульти-лейбл классификации
rf_model = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=42)
rf_model.fit(X_train, y_train)

# Получение предсказаний для валидационной выборки и вычисление multilabel f1-score
y_pred = rf_model.predict(X_val)
f1score = f1_score(y_val, y_pred, average='samples')

# Выводим multilabel f1-score
print("Multilabel f1-score:", f1score)

# Получение предсказаний для тестовой выборки
X_test_tfidf = vectorizer.transform(test_data['text'])
test_predictions = rf_model.predict(X_test_tfidf)

# Создание нового датафрейма для предсказанных меток
result_df = pd.DataFrame(test_predictions, columns=labels.columns)

# Создание столбца 'labels' на основе значений столбцов классов'
result_df['labels'] = result_df.apply(lambda row: "{" + ', '.join([col for col in labels.columns if row[col]==1]) + "}", axis=1)

# Сохранение предсказанных классов в файл
result_df['labels'].to_csv('predicted_labels.csv', index=True)