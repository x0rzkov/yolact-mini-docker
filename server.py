import os
import sys
import subprocess
import requests
import ssl
import random
import string
import json
import logging
from logging.handlers import RotatingFileHandler

from flask import redirect
from flask import url_for
from flask import jsonify
from flask import Flask
from flask import request
from flask import send_file
from flask import Response
from flask import abort
from flask import send_from_directory
from flask import render_template_string

import traceback
from werkzeug.utils import secure_filename

try:
    from PIL import Image
except ImportError:
    import Image

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
    from io import BytesIO

import torch
import torch.backends.cudnn as cudnn
import argparse
import glob
import cv2
import time

from modules.build_yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils.functions import MovingAverage, ProgressBar
from utils import timer
from data.config import update_config
from utils.output_utils import NMS, after_nms, draw_img

try:  # Python 3.5+
    from http import HTTPStatus
except ImportError:
    try:  # Python 3
        from http import client as HTTPStatus
    except ImportError:  # Python 2
        import httplib as HTTPStatus

UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = set(['png', 'jpeg', 'jpg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def download_file(url, filename=None):
    if filename is None:
        local_filename = url.split('/')[-1]
    else:
        local_filename = filename
    r = requests.get(url)
    destFile = os.path.join(app.config['UPLOAD_FOLDER'], local_filename)
    f = open(destFile, 'wb')
    for chunk in r.iter_content(chunk_size=512 * 1024): 
        if chunk:
            f.write(chunk)
    f.close()
    return destFile

@app.route("/process", methods=["POST", "GET"])
def process():
    try:
        destFile = ""
        if request.method == 'POST':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                destFile = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(destFile)
                app.logger.warning('filename=(%s)', filename)
        else:
            app.logger.warning("Request dictionary data: {}".format(request.data))
            app.logger.warning("Request dictionary form: {}".format(request.form))
            url = request.form["url"]
            print("url:", url)
            # download file
            destFile = download_file(url)

        # app.logger.error('An error occurred')
        app.logger.warning('destFile=(%s)', destFile)

        img_name = destFile.split('/')[-1]
        app.logger.warning('img_name=(%s)', img_name)

        img_origin = torch.from_numpy(cv2.imread(destFile)).float()
        if cuda:
            img_origin = img_origin.cuda()
        img_h, img_w = img_origin.shape[0], img_origin.shape[1]
        img_trans = FastBaseTransform()(img_origin.unsqueeze(0))
        net_outs = net(img_trans)
        nms_outs = NMS(net_outs, args.traditional_nms)

        app.logger.warning('img_h=(%s)', img_h)
        app.logger.warning('img_w=(%s)', img_w)

        app.logger.warning('cuda=(%s)', cuda)
        app.logger.warning('args.show_lincomb=(%s)', args.show_lincomb)
        app.logger.warning('args.no_crop=(%s)', args.no_crop)
        app.logger.warning('args.visual_thre=(%s)', args.visual_thre)
        app.logger.warning('args=(%s)', args)

        show_lincomb = bool(args.show_lincomb)
        with timer.env('after nms'):
            results = after_nms(nms_outs, img_h, img_w, show_lincomb=show_lincomb, crop_masks=not args.no_crop,
                                visual_thre=args.visual_thre, img_name=img_name)
            if cuda:
                torch.cuda.synchronize()

        # app.logger.warning('results=(%s)', results)
        img_numpy = draw_img(results, img_origin, args)

        cv2.imwrite(f'results/images/{img_name}', img_numpy)
        # print(f'\r{i + 1}/{num}', end='')

        try:
            im = Image.open(f'results/images/{img_name}')
            # im = Image.open(destFile)
            io = BytesIO()
            im.save(io, format='JPEG')
            return Response(io.getvalue(), mimetype='image/jpeg')

        except IOError:
            abort(404)

        # return send_from_directory('.', filename), 200
        callback = json.dumps({"results": results})
        return callback, 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400


if __name__ == '__main__':
    global net, cuda, args

    # Load model 
    parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model', default='weights/yolact_base_54_800000.pth', type=str)
    parser.add_argument('--visual_top_k', default=100, type=int, help='Further restrict the number of predictions to parse')
    parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
    parser.add_argument('--hide_mask', default=False, action='store_true', help='Whether to display masks')
    parser.add_argument('--hide_bbox', default=False, action='store_true', help='Whether to display bboxes')
    parser.add_argument('--hide_score', default=False, action='store_true', help='Whether to display scores')
    parser.add_argument('--show_lincomb', default=False, action='store_true',
                        help='Whether to show the generating process of masks.')
    parser.add_argument('--no_crop', default=False, action='store_true',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str, help='The folder of images for detecting.')
    parser.add_argument('--video', default=None, type=str,
                        help='The path of the video to evaluate. Pass a number to use the related webcam.')
    parser.add_argument('--real_time', default=False, action='store_true', help='Show the detection results real-timely.')
    parser.add_argument('--visual_thre', default=0.3, type=float,
                        help='Detections with a score under this threshold will be removed.')

    args = parser.parse_args()
    strs = args.trained_model.split('_')
    config = f'{strs[-3]}_{strs[-2]}_config'

    update_config(config)
    print(f'\nUsing \'{config}\' according to the trained_model.\n')

    with torch.no_grad():
        cuda = torch.cuda.is_available()
        if cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        net = Yolact()
        net.load_weights('weights/' + args.trained_model, cuda)
        net.eval()
        print('Model loaded.\n')

        if cuda:
            net = net.cuda()

    port = 8000
    host = '0.0.0.0'

    handler = RotatingFileHandler('yolact.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)

    app.run(host=host, port=port, threaded=False)
