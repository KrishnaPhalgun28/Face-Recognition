# Face-Recognition
#MTCNN #Keras #VGGFace #Open-CV #Scipy

## Installation

```sh
   pip install -r requirements.txt
```

## Usage

- To train the model
  - Create a folder that has a face label as the name in the train directory
  - Save images that contain only one face in that folder
  - `python detect.py -t`

- To test the model
   - to recognize faces in a live video `python detect.py -l`
   - to recognize faces in an image file `python detect.py -p ${path_to_image}`

## Example

Execute

```sh
   python detect.py -p test\Elon_Musk_Wiki.jfif
```

Output

![model output](output.jpeg)
