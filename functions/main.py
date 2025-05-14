import base64
import cv2
import numpy as np
from firebase_functions import https_fn
from firebase_admin import initialize_app

initialize_app()

@https_fn.on_request()
def denoise_image(req: https_fn.Request) -> https_fn.Response:
    try:
        data = req.get_json()
        image_b64 = data.get("image")
        if not image_b64:
            return ("No image provided", 400)

        image_data = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Denoising
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        _, buffer = cv2.imencode(".jpg", denoised)
        denoised_b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "denoised_image": denoised_b64
        }
    except Exception as e:
        return ({
            "error": str(e)
        }, 500)