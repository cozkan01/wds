I did this project because OmniParser sometimes misses small icons or text on the screen. Enhanced via Laplacian Structure Detection (LSD) 
### How to use this project
You need to have Node.js and Python in your computer.

1. First,:
```bash
npm install
```

2. The models are very big, so we don't put them in github. You must download them yourself using our script:
```bash
cd OmniParser
pip install -r requirements.txt
python download_models.py
```

3. Start the application:
```bash
npm run tauri dev
```

### Comparison Images
Comparison between OmniParser and Hybrid WDS: (Blue is OmniParser, Red is Hybrid WDS)

![Comparison](1.png)

![Comparison](2.png)

![Comparison](3.png)
