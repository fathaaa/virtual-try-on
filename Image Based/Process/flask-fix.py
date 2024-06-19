from ultralytics import YOLO
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)