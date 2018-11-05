from init_flask import db, app
from recognizer.train import train


if __name__ == '__main__':
    train()
    app.run(debug=False)
