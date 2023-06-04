import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Загрузка датасета из файла
dataset_path = 'dataset.csv'
dataset = pd.read_csv(dataset_path)

# Подготовка данных
X = dataset['фразы'].values
y = dataset['группы'].values

# Преобразование текстовых данных в последовательности токенов
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_encoded = tokenizer.texts_to_sequences(X)

max_sequence_length = max(len(seq) for seq in X_encoded)
X_padded = pad_sequences(X_encoded, maxlen=max_sequence_length)

# Преобразование целевых строковых значений в числовые метки классов
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Преобразование целевых значений в формат one-hot encoding
num_classes = len(label_encoder.classes_)
y_one_hot = to_categorical(y_encoded, num_classes=num_classes)

# Создание модели нейронной сети
model = Sequential()

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(64))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model.fit(X_padded, y_one_hot, epochs=10, batch_size=32)

# Тесты
X_test = ['Какая классная погода', 'Где ты этому научился?','Это моя любимая песня','Сколько тебе лет?','Какой у тебя любимый фильм?','Какая твоя любимая книга?','Когда ты последний раз ел?','Есть ли у тебя животное?']
X_test_encoded = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_encoded, maxlen=max_sequence_length)
predictions = model.predict(X_test_padded)

decoded_predictions = label_encoder.inverse_transform(predictions.argmax(axis=1))
print(decoded_predictions)
