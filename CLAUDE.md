# InsTaG Framework Commands and Guidelines

## Common Commands
- **Build Environment**: `conda env create --file environment.yml`
- **Process Video**: `python data_utils/process.py data/<ID>/<ID>.mp4`
- **Generate Teeth Mask**: `python data_utils/easyportrait/create_teeth_mask.py ./data/<ID>`
- **Extract Audio Features**: `python data_utils/deepspeech_features/extract_ds_features.py --input data/<n>.wav`
- **Pre-training**: `bash scripts/pretrain_con.sh data/pretrain output/<project_name> <GPU_ID>`
- **Fine-tuning**: `bash scripts/train_xx_few.sh data/<ID> output/<project_name> <GPU_ID>`
- **Synthesis**: `python synthesize_fuse.py -S data/<ID> -M output/<project_name> --audio <path> --audio_extractor <type>`
- **Docker Commands**: Use `./docker-run.sh` with various subcommands (see README_docker.md)

## Code Style Guidelines
- **Python Version**: 3.9 for main code, 3.10 for Sapiens
- **Formatting**: Follow existing style in files (indentation, line breaks)
- **Imports**: Group standard library, third-party, and local imports
- **Naming**: Use snake_case for variables/functions, CamelCase for classes
- **Error Handling**: Use try/except blocks for file operations and external calls
- **Documentation**: Add docstrings for new functions and classes

## Project Structure
- `/data`: Input videos and processed data
- `/output`: Generated models and results
- `/data_utils`: Processing utilities for various modalities
- `/scene`: Core rendering and modeling code
- `/utils`: Helper functions for audio, image, and graphics processing