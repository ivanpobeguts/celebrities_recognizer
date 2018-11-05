from flask import Flask, render_template, request, redirect, url_for
from _datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import numpy
from recognizer.settings import logger
from recognizer.recognize import recognize
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///celebrities.sqlite3'
app.config['SECRET_KEY'] = "random string"
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'people')

db = SQLAlchemy(app)


class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    photos = db.relationship('Photo', backref='person', lazy=True)


class Photo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    img = db.Column(db.String(100000), nullable=False)
    person_id = db.Column(db.Integer, db.ForeignKey('person.id'),
                          nullable=False)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            logger.info('No file part')
            return redirect(request.url)
        file = request.files['file'].read()
        logger.info(file)
        npimg = numpy.fromstring(file, numpy.uint8)
        result = recognize(npimg)
        full_filename = os.path.join('static', 'people', 'out.jpg')
        return redirect(url_for('index', full_filename=full_filename))
    else:
        full_filename = request.args.get('full_filename')
        now = datetime.now()
        return render_template('index.html', user_image=full_filename, date=now)


@app.route('/celebrities')
def celebrities():
    return render_template('celebrities.html', celebrities=Person.query.all())
