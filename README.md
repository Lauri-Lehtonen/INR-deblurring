# INR-deblurring
Code implementation for my B.Sc thesis IMPLICIT NEURAL REPRESENTATIONS FOR NON-BLIND DEPTH-AWARE IMAGE DEBLURRING https://urn.fi/URN:NBN:fi:tuni-202405306517

Built on ParallaxICB by Torres and Kämäräinen, which can be found at https://github.com/germanftv/ParallaxICB/tree/main


1. Installation
    
    Clone repository
    
        git clone https://github.com/Lauri-Lehtonen/INR-deblurring
        cd INR-deblurring
    
    Create conda environment
    
        conda create -y --name INR-deblurring python=3.8 && conda activate INR-deblurring
    
    Install dependencies
    
        pip install -r requirements.txt

3. Datasets

    Download and unzip datasets from https://github.com/germanftv/ParallaxICB/tree/main

4. Configuration file: ./configs/config.yaml

    Change the root directories (VirtualCMB and RealCMB) to the paths where the datasets are.

5. Info files

    Run the following instruction to generate dataset info files:
    
        python setup.py
    
6. InstantNGP (refer to https://github.com/NVlabs/tiny-cuda-nn/tree/master if issues arise)
   
    Download the InstantNGP repository and compile it

        git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
        cd tiny-cuda-nn
       
    Then, use CMake to build the project: (on Windows, this must be in a developer command prompt)
    
        cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
        cmake --build build --config RelWithDebInfo -j

    Then install the pytorch extention by running the following commands.
    
        cd bindings/torch
        python setup.py install

7. Running of script.

    experiments can be run with the command:
    
        python main.py --dataset $DATASET --model $BLUR_MODEL --id $ID --nn_model $MODELNAME
    
    where:
    
        $DATASET is the experimental dataset. Options: RealCMB or VirtualCMB.
        $BLUR_MODEL is the blur model to be tested. Options: ICB (proposed model) or PWB (baseline).
        $ID is the image id in the dataset info file. Options: 0-57 for RealCMB, or 0-982 for VirtualCMB.
        $MODELNAME is the name of the blur model to be used. Options: SIREN, FOURIER_MAPPED_MLP, HASH_ENCODING, DICTIONARY_FIELD
    
    For instance: python main.py blur --dataset RealCMB --model ICB --id 0 --nn_model HASH_ENCODING
    
    To run entire dataset at once use the run_dataset.sh script:
    
        run_dataset main.py deblur $MODEL $DATASET $MODELNAME $START_ID(indexing starts at 0) $END_ID
    
    To run summary for all tests run the following script:
    
        python summary.py --dataset $DATASET --model $BLUR_MODEL 
