# Movenet Pose estimation tensorflow 

## Usage

### Clone the repo or download to the local machine.
### Download models from tensorflow hub
Open terminal/command prompt in the root directory of this repo and do``` bash install.sh ```.

### Run the file
- #### To use lightning model
```
python run.py --m models/lightning.tflite -f mapping.json
```
- #### To use thunder model
```
python run.py --m models/thunder.tflite -f mapping.json
```