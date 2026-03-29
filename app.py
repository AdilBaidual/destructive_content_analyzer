import os
import time
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory

from config import STATIC_DIR, RAW_DATA_DIR, PREPARED_DATA_DIR
from parser.tg_parser import parse_telegram_channel, parse_posts_by_ids, parse_ids_string, load_raw_data
from preprocessing.preprocessor import process_and_save
from analysis.analyzer import analyze_file
from analysis.extremism_checker import check_extremism_file
from models.trainer import retrain_destructive_model, retrain_extremism_model, check_models_exist, train_all_models

app = Flask(__name__)
app.secret_key = os.urandom(24)

for dir_path in [STATIC_DIR, RAW_DATA_DIR, PREPARED_DATA_DIR]:
    os.makedirs(dir_path, exist_ok=True)


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    filename = None
    extremism_result = None

    if request.method == 'POST':
        channel = request.form.get('channel', '').strip()
        count = request.form.get('count', '100')

        try:
            count = int(count)
            models_status = check_models_exist()

            if not models_status['destructive']:
                error = "Модель не обучена. Запустите: python models/trainer.py"
            else:
                raw_filename = parse_telegram_channel(channel, count)
                prepared_filename = process_and_save(raw_filename)
                result = analyze_file(prepared_filename)
                filename = prepared_filename
                session['last_filename'] = prepared_filename

        except Exception as e:
            error = f"Ошибка: {str(e)}"

    return render_template('index.html',
                          result=result, error=error, filename=filename,
                          extremism_result=extremism_result, now=int(time.time()))


@app.route('/check_extremism', methods=['POST'])
def check_extremism():
    filename = request.form.get('filename') or session.get('last_filename')
    top_n = int(request.form.get('top_n', 10))

    if not filename:
        return redirect(url_for('index'))

    try:
        models_status = check_models_exist()
        if not models_status['extremism']:
            error = "Модель экстремизма не обучена"
            return render_template('index.html', error=error, now=int(time.time()))

        result = analyze_file(filename)
        extremism_result = check_extremism_file(filename, top_n=top_n)

        return render_template('index.html',
                              result=result, extremism_result=extremism_result,
                              filename=filename, now=int(time.time()))
    except Exception as e:
        return render_template('index.html', error=str(e), now=int(time.time()))


@app.route('/retrain', methods=['GET'])
def retrain_page():
    return render_template('retrain.html', step=1)


@app.route('/retrain/load', methods=['POST'])
def retrain_load():
    channel = request.form.get('channel', '').strip()
    post_ids_str = request.form.get('post_ids', '').strip()
    model_type = request.form.get('model_type', 'destructive')

    if not channel or not post_ids_str:
        return render_template('retrain.html', step=1,
                              message="Укажите канал и ID постов", message_type="warning")

    try:
        post_ids = parse_ids_string(post_ids_str)

        if not post_ids:
            return render_template('retrain.html', step=1,
                                  message="Не удалось распознать ID", message_type="warning")

        raw_filename = parse_posts_by_ids(channel, post_ids)
        df = load_raw_data(raw_filename)
        posts = df.to_dict('records')

        session['retrain_filename'] = raw_filename
        session['retrain_model_type'] = model_type

        return render_template('retrain.html', step=2, posts=posts, model_type=model_type)

    except Exception as e:
        return render_template('retrain.html', step=1,
                              message=f"Ошибка: {str(e)}", message_type="danger")


@app.route('/retrain/train', methods=['POST'])
def retrain_train():
    model_type = request.form.get('model_type', 'destructive')
    posts_count = int(request.form.get('posts_count', 0))

    if posts_count == 0:
        return render_template('retrain.html', step=1,
                              message="Нет данных", message_type="warning")

    try:
        texts = []
        labels = []

        for i in range(posts_count):
            text = request.form.get(f'post_{i}_text', '')
            label = request.form.get(f'post_{i}_label', '')

            if text and label:
                texts.append(text)
                labels.append(int(label))

        if not texts:
            return render_template('retrain.html', step=1,
                                  message="Нет размеченных данных", message_type="warning")

        if model_type == 'extremism':
            result = retrain_extremism_model(texts, labels)
        else:
            result = retrain_destructive_model(texts, labels)

        return render_template('retrain.html', step=3, result=result,
                              message="Модель дообучена!", message_type="success")

    except Exception as e:
        return render_template('retrain.html', step=1,
                              message=f"Ошибка: {str(e)}", message_type="danger")


@app.route('/train_models', methods=['GET'])
def train_models_page():
    try:
        results = train_all_models()
        return render_template('retrain.html', step=3,
                              result={'new_examples': 'все', 'final_accuracy': results['destructive']['accuracy']},
                              message="Модели обучены!", message_type="success")
    except Exception as e:
        return render_template('retrain.html', step=1,
                              message=f"Ошибка: {str(e)}", message_type="danger")


if __name__ == '__main__':
    print("=" * 50)
    print("Анализатор деструктивного контента")
    print("=" * 50)

    models_status = check_models_exist()
    print(f"Модель деструктива: {'✓' if models_status['destructive'] else '✗'}")
    print(f"Модель экстремизма: {'✓' if models_status['extremism'] else '✗'}")

    if not all(models_status.values()):
        print("\nДля обучения: python models/trainer.py")

    print(f"\nСервер: http://127.0.0.1:5000")
    print("=" * 50)

    app.run(debug=True, port=5000)
