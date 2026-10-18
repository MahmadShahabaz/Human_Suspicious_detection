from flask import Flask, render_template, url_for, request, Response, jsonify
import sqlite3
import os
from werkzeug.utils import secure_filename
from stream_engine import generate_frames, get_new_stream_token, stop_all_streams

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form.get('name', '')
        password = request.form.get('password', '')

        query = "SELECT name, password FROM user WHERE name = ? AND password = ?"
        cursor.execute(query, (name, password))
        result = cursor.fetchall()
        connection.close()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided, Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form.get('name', '')
        password = request.form.get('password', '')
        mobile = request.form.get('phone', '')
        email = request.form.get('email', '')

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES (?, ?, ?, ?)", (name, password, mobile, email))
        connection.commit()
        connection.close()

        return render_template('index.html', msg='Successfully Registered')

    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route that yields multipart JPEG frames."""
    module = request.args.get('module', 'weapon')
    source = request.args.get('source', '0')
    token = get_new_stream_token()
    return Response(
        generate_frames(module, source, token),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/stop_stream', methods=['GET', 'POST'])
def stop_stream():
    """Stops any currently active detection stream and releases resources."""
    stop_all_streams()
    return jsonify({'status': 'stopped', 'message': 'All detection streams stopped and resources released'})


@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handles video file uploads from dashboard forms."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    relative_path = os.path.join('uploads', filename)
    return jsonify({
        'status': 'success',
        'filename': filename,
        'path': relative_path
    })


# Legacy form POST routes - now seamlessly start detection inside the dashboard without subprocess windows
@app.route('/video', methods=['GET', 'POST'])
def video():
    if request.method == 'POST':
        filename = request.form.get('filename', '')
        source = ('test/' + filename) if filename else 'test/test1.mp4'
        return render_template('userlog.html', initial_module='analyse', initial_source=source)
    return render_template('userlog.html')


@app.route('/Live1', methods=['GET', 'POST'])
def Live1():
    if request.method == 'POST':
        filename = request.form.get('filename1', '')
        source = ('testweapon/' + filename) if filename else '0'
        return render_template('userlog.html', initial_module='weapon', initial_source=source)
    return render_template('userlog.html')


@app.route('/Live2', methods=['GET', 'POST'])
def Live2():
    if request.method == 'POST':
        filename = request.form.get('filename2', '')
        source = ('testaccident/' + filename) if filename else '0'
        return render_template('userlog.html', initial_module='accident', initial_source=source)
    return render_template('userlog.html')


@app.route('/Live3', methods=['GET', 'POST'])
def Live3():
    if request.method == 'POST':
        filename = request.form.get('filename3', '')
        source = ('testexplosion/' + filename) if filename else '0'
        return render_template('userlog.html', initial_module='explosion', initial_source=source)
    return render_template('userlog.html')


@app.route('/Live4', methods=['GET', 'POST'])
def Live4():
    if request.method == 'POST':
        filename = request.form.get('filename4', '')
        source = ('testperson/' + filename) if filename else '0'
        return render_template('userlog.html', initial_module='person', initial_source=source)
    return render_template('userlog.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
