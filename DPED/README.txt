1. Prerequisites:

    python + python libraries [tensorflow (1.0.1 and above), numpy, scipy]


2. To run pre-trained models, use the following command:

    python run_model.py model=<model> resolution=<resolution> use_gpu=<use_gpu>

    where <model> is the corresponding phone model: {iphone, sony, blackberry}
    <resolution>: {orig, high, medium, small, tiny}
    <use_gpu>: {true, false}

    example:

    python run_model.py model=iphone resolution=orig use_gpu=true


3. Can I run the models on CPU?

    Yes, you can set the parameter <use_gpu> to false, but note that in this case the performance will be low (up to 3-5 minutes per image)


4. What if I get an error: "OOM when allocating tensor with shape[...]"?

    Your GPU does not have enough memory to process full-resolution images (this usually happens when it has less than 6GB or RAM)

    Solutions:

    a) Run the model on CPU (set the parameter <use_gpu> to false). Note that this will take considerably more time (3-5 minutes per image)

    b) Use cropped images, set the parameter <resolution> to:

    high    - center crop of size 1680x1260 pixels
    medium  - center crop of size 1366x1024 pixels
    small   - center crop of size 1024x768 pixels
    tiny    - center crop of size 800x600 pixels

    The less resolution is - the smaller part of the image will be processed

5. Folder structure:

    models/                 - pre-trained models for three phones (iphone, sony and blackberry)
    test_photos/blackberry/ - test photos for blackberry phone
    test_photos/iphone/     - test photos for iphone
    test_photos/sony/       - test photos for sony phone
    results/                - processed images will appear in this folder
    run_model.py            - python script that loads test images and applies pre-trained models to them
    model.py                - python script that specifies the architecture of the image enhancement network (resnet)
    utils.py                - python script for command arguments / input images processing

6. Any other questions / problems?

    You can contact andrey.ignatoff@gmail.com for more information
