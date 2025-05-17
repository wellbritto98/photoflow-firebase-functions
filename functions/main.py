import base64
import cv2
import numpy as np
from firebase_functions import https_fn
from firebase_admin import initialize_app

initialize_app()

@https_fn.on_request()
def denoise_image(req: https_fn.Request) -> https_fn.Response:
    try:
        # Verifica se o arquivo foi enviado
        if 'img' not in req.files:
            return ("No image file provided", 400)

        file = req.files['img']
        in_memory_file = file.read()
        nparr = np.frombuffer(in_memory_file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Denoising
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 8, 7, 21)

        # Sharpening filter
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        _, buffer = cv2.imencode(".jpg", sharpened)

        # Retorna como base64 (pois não há send_file)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        return {
            "denoised_image": img_b64
        }
    except Exception as e:
        return ({
            "error": str(e)
        }, 500)