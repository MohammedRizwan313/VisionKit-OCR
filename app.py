from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tempfile
import os
import pathlib
import Quartz
import Vision
from Cocoa import NSURL
from Foundation import NSDictionary
from wurlitzer import pipes

app = FastAPI()

def image_to_text(img_path, lang="eng"):
    input_url = NSURL.fileURLWithPath_(img_path)

    with pipes() as (out, err):
        input_image = Quartz.CIImage.imageWithContentsOfURL_(input_url)

    vision_options = NSDictionary.dictionaryWithDictionary_({})
    vision_handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
        input_image, vision_options
    )
    results = []
    handler = make_request_handler(results)
    vision_request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)
    error = vision_handler.performRequests_error_([vision_request], None)

    return results

def make_request_handler(results):
    """ results: list to store results """
    if not isinstance(results, list):
        raise ValueError("results must be a list")

    def handler(request, error):
        if error:
            print(f"Error! {error}")
        else:
            observations = request.results()
            for text_observation in observations:
                recognized_text = text_observation.topCandidates_(1)[0]
                results.append([recognized_text.string(), recognized_text.confidence()])
    return handler

@app.post("/detect_text/")
async def detect_text(image: UploadFile = File(...)):
    try:
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(await image.read())
            temp_path = temp.name

        # Process the image and extract text
        results = image_to_text(temp_path)

        # Convert the results into a string
        extracted_text = "\n".join([result[0] for result in results])

        return {"text": extracted_text}
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
