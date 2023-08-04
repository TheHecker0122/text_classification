from tensorflow.keras.models import load_model

loaded_model = load_model('my_model.h5')

# Тесты
X_test = ['Какая классная погода', 'Где ты этому научился?','Это моя любимая песня','Сколько тебе лет?','Какой у тебя любимый фильм?','Какая твоя любимая книга?','Когда ты последний раз ел?','Есть ли у тебя животное?']
X_test_encoded = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_encoded, maxlen=max_sequence_length)
predictions = loaded_model.predict(X_test_padded)

decoded_predictions = label_encoder.inverse_transform(predictions.argmax(axis=1))
print(decoded_predictions)

'''
Загрузка  уже обученой модели
'''
