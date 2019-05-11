# Joint Learning of Facial Expression and Head Pose from Speech

Using the joint modalities model - where we predict the face deformation
from the audio - then the rigid head pose from the face deformation.

paper: <https://www.isca-speech.org/archive/Interspeech_2018/pdfs/2587.pdf>


## Install

A conda environment file is provided - `env.yml` with the requirements.


## Testing

To view a sample prediction run:

    python wav2speech.py

## Rendering

A render utility is provided, so the json output can be viewed as a movie.

    python render.py scene_a_01_0184_pred.json scene_a_01_0184_pred.mp4

The resulting movie should be muxed with the corresponding wav file.
