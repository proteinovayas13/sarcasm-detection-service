# Sentiment - analysis

Самая быстрая модель `Logistic Regression + TF-IDF (Classic ML)`, но предсказывает плохо, для инференса берем "BERT-Tiny " обученную на небольшом корпусе из 28619 текстов на английском языке
(модель для предсказаний находится в src/models
/predict_model.py).

## Code

Для обучения используется обычный Trainer.

Для развертывания модели используется Pipeline и FastAPI, стандартный порт `1234`.

## Expertiments

Данные:
- дисбаланса нет

Что использовалось:
- 🔵 Logistic Regression + TF-IDF (Classic ML)
- 🐱 CatBoost + SentenceTransformer (Ensemble + Embeddings)
- 🤖 BERT-Tiny (Transformer-based)
- 🦾 RoBERTa-Base (Transformer-based)

Предобработка и валидация:
- По стандарту делим датасет на отложенную и обучающую в соотношении `0.8/0.2`.
- В языковых моделях используем всю выборку и проверяем уже на тестовой при валидации в трейнере.
- В качестве метрики - `F1`и Accuracy.

## Запуск:

-Собрать образ из папки docker
docker build -t sarcasm-service -f docker/Dockerfile .

Или если Dockerfile в корне docker папки
docker build -t sarcasm-service docker/

Запустить контейнер
docker run -p 8000:8000 sarcasm-service

Открыть http://localhost:8000/docs

Или использовать curl команды

## Deploy and inference

Собранный образ основан на базовом Python. Новый образ нужно собрать с помощью `build`.

Для инференса используем Docker-compose:

```bash
docker compose up --build -d
```

После чего можно споконо заходить на Swagger через порт `1234`.

    ## TODO

- [ ] Аугментация на основе текста без скобок
- [ ] Предочистка мусора во время инференса
- [ ] Оптимизация модели
- [ ] Автоматический подбор гиперпараметров
- [ ] Docker для обучения-экспериментов и инференса итоговой модели
- [ ] Переписать код на нативный PyTorch
