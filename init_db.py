from init_flask import db, Person, Photo
import base64
import os


def init_db():
    db.drop_all()
    db.create_all()
    main_path = 'samples/small_sample'
    dirs = os.listdir(main_path)
    print(dirs)
    for dir_name in dirs:
        # print(dir_name)
        if os.path.isdir(os.path.join(main_path, dir_name)):
            person = Person(name=dir_name)
            print(person)
            images = os.listdir(os.path.join(main_path, dir_name))
            for image_path in images:
                full_path = os.path.join(main_path, dir_name, image_path)
                with open(full_path, "rb") as image_file:
                    img_byte = base64.b64encode(image_file.read())
                photo = Photo(img=img_byte)
                person.photos.append(photo)
                db.session.add(photo)
            db.session.add(person)
        # commit
    db.session.commit()


if __name__ == '__main__':
    init_db()
    print(Person.query.all())
