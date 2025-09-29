# Дипломный Ассистент 🤖🎓

RAG-ассистент, отвечающий на вопросы по моей дипломной работе) (./data/diplom.pdf). Отвечает на вопросы по содержанию диплома, используя передовые технологии поиска и генерации текста.

---

## 🔧 Стек технологий
- **Backend**: Python 3.10+, LangChain, LangGraph
- **Векторная БД**: Qdrant (Docker)
- **LLM**: Mistral AI (`mistral-large-latest`)
- **Embeddings**: `intfloat/multilingual-e5-base`
- **Загрузка данных**: PyPDFLoader + текстовый сплиттинг

---

## 🚀 Установка и запуск

### 1. Клонируйте репозиторий и установите зависимости:
```bash
git clone <репозиторий>
cd diplom-assistant
pip install -r requirements.txt
```

### 2. Настройте окружение:
Создайте `.env` файл с переменными:
```env
MISTRAL_API_KEY=your_mistral_key
QDRANT_URL=http://localhost:6333
```

### 3. Запустите Qdrant в Docker:
```bash
docker-compose up -d
```

### 4. Запустите ассистента:
```bash
python main.py
```

---

## 📂 Структура проекта
```
.
├── data/                  # PDF-файлы диплома
├── services/
│   ├── qdrant.py          # Работа с векторной БД
│   ├── llm.py             # Интеграция с Mistral AI
│   └── loader.py          # Загрузка и обработка PDF
├── config/                # Конфигурации (settings.py)
├── states/                # Определение состояний графа
├── docker-compose.yml     # Конфиг Qdrant
└── main.py                # Точка входа
```

---

## 💡 Примеры кода

### 1. Загрузка PDF и разбивка на чанки
```python
from services.loader import Loader

loader = Loader()
loader.load_local_pdf()  # Загружает все PDF из ./data/pdf/
```

### 2. Поиск контекста в Qdrant
```python
from services.qdrant import QdrantService

qdrant = QdrantService()
context_docs = qdrant.retrive("Что такое RAG?")  # Возвращает список Document
```

### 3. Генерация ответа с LLM
```python
from services.llm import LLMService
from states.graph_states import State

llm = LLMService()
state = State(question="Объясни архитектуру проекта", context=context_docs)
response = llm.generate(state)
print(response.content)
```

### 4. Пример графа выполнения (LangGraph)
```python
from langgraph.graph import StateGraph

def build_graph():
    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)  # Поиск контекста
    graph.add_node("generate", generate)  # Генерация ответа
    graph.add_edge("retrieve", "generate")
    graph.set_entry_point("retrieve")
    return graph.compile()
```

---

## 🔄 Как работает система?
1. **Загрузка данных**: PDF разбивается на чанки (200 символов с перекрытием 50).
2. **Векторизация**: Текст конвертируется в embeddings и сохраняется в Qdrant.
3. **Поиск**: Для вопроса ищутся топ-10 релевантных чанков.
4. **Генерация**: Mistral AI формирует ответ на основе контекста.

---
## 📌 Пример вопроса/ответа
**Вопрос**:
*"Какие технологии использовались в дипломной работе?"*

**Ответ**:
> "В проекте применялись:
> - **LangChain** для оркестрации RAG-пайплайна,
> - **Qdrant** для хранения и поиска векторных embeddings,
> - **Mistral AI** (`mistral-large-latest`) для генерации ответов.
> Подробности смотрите в разделе 3.2 архитектуры (стр. 15)."
```

---
**⚠️ Важно**:
- Убедитесь, что PDF-файл диплома лежит в `./data/pdf/`.
- Для работы Mistral AI нужен действующий API-ключ ([получить здесь](https://mistral.ai/)).
- Qdrant автоматически развернётся через Docker при первом запуске.
--------------------------