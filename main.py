from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os


def load_data(folder_path):
    files = os.listdir(folder_path)
    reviews = []
    for file in files:
        with open(folder_path + '\\' + file) as f:
            reviews.append(f.read())
    return reviews


def generate_tf_idf():
    pos_revs = load_data('D:\\python-projects\\NLP-ASSIGNMENT2\\txt_sentoken\\pos')
    neg_revs = load_data('D:\\python-projects\\NLP-ASSIGNMENT2\\txt_sentoken\\neg')
    revs = pos_revs + neg_revs
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(revs).toarray()
    return vectors


def create_model():
    labels = []
    for i in range(0, 2000):
        if i < 1000:
            labels.append(1)
        else:
            labels.append(0)
    data = generate_tf_idf()
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = LogisticRegression(random_state=0).fit(x_train, y_train)
    joblib.dump(model, 'model.pkl')
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)
    return train_acc, test_acc


def ui():
    print('Enter your choice:\n')
    choice = input('1) Enter review\n2) Load from file\n')
    review = None
    if choice == '1':
        review = input('Insert your review: ')
    elif choice == '2':
        path = input('Enter file path: ')
        with open(path) as file:
            review = file.read()
    pos_revs = load_data('D:\\python-projects\\NLP-ASSIGNMENT2\\txt_sentoken\\pos')
    neg_revs = load_data('D:\\python-projects\\NLP-ASSIGNMENT2\\txt_sentoken\\neg')
    revs = pos_revs + neg_revs
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit_transform(revs).toarray()
    vector = vectorizer.transform([review]).toarray()
    model = joblib.load('model.pkl')
    if model.predict(vector) == 1:
        print('Positive')
    else:
        print('Negative')


# create_model()
ui()
