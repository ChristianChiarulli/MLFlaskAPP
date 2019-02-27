# Running ML Flask app

First you need miniconda or a way to create a python virtual environment

You can download miniconda into your current directory like so:

```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
``` 

Then run it, just agree to defaults and say `yes` when it asks

```sh
sh miniconda.sh
```

After installatiion source your bashrc

```sh
source .bashrc
```

now we can create a virtual environment

```sh
conda create -n flask python=3.6
```

now we can activate it

```sh
source activate flask
```

We are ready to install some necessary packages

```sh
pip install Pillow, tensorflow, keras, flask
```

Now we need to set an environment variable for flask

```sh
export FLASK_APP=predict_app.py
```

You can now run the app (make sure your in the same directory as predict.py)

```sh
flask run --host=0.0.0.0
```

Now navigate to `localhost:5000/static/predict.html`
