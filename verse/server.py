from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64
import json
from OptimalPath import ImageSeg, OptimalPathing  # Import your classes from OptimalPath

PORT = 8000

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            # Decode the image data and save it to a file
            img_data = BytesIO(post_data)
            img = Image.open(img_data)
            img_np = np.array(img)

            # Process the image and compute optimal path
            obj = ImageSeg(img_np)
            processed_img = obj.IsoGrayThresh()
            obj1 = OptimalPathing(processed_img)
            plot_image, processed_image = obj1.ComputeAStar()

            # Prepare the response
            if plot_image is not None and processed_image is not None:
                # Convert images to base64
                plot_img_buffer = BytesIO()
                plt.imsave(plot_img_buffer, plot_image, format='png')
                plot_img_str = base64.b64encode(plot_img_buffer.getvalue()).decode('utf-8')

                processed_img_buffer = BytesIO()
                plt.imsave(processed_img_buffer, processed_image, format='png')
                processed_img_str = base64.b64encode(processed_img_buffer.getvalue()).decode('utf-8')

                # Send the CORS headers in the response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', 'http://127.0.0.1:5500')  # Allow requests from specific origin
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')  # Allow specific methods
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')  # Allow specific headers
                self.send_header('Access-Control-Allow-Credentials', 'true')  # Allow credentials (cookies, authorization headers)
                self.end_headers()

                # Send the JSON response with the images
                response_data = {
                    'plot_image': plot_img_str,
                    'processed_image': processed_img_str
                }
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            else:
                # Handle error if images are None
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', 'http://127.0.0.1:5500')  # Allow requests from specific origin
                self.end_headers()
                error_message = {'error': 'Unable to process images.'}
                self.wfile.write(json.dumps(error_message).encode('utf-8'))
        except Exception as e:
            # Handle any exceptions and send an error response
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', 'http://127.0.0.1:5500')  # Allow requests from specific origin
            self.send_header('Access-Control-Allow-Credentials', 'true')  # Allow credentials (cookies, authorization headers)
            self.end_headers()
            error_message = {'error': str(e)}
            self.wfile.write(json.dumps(error_message).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=RequestHandler, port=PORT):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
