# PyCausalSim Interactive Demo

A web-based interactive demo for [PyCausalSim](https://github.com/Bodhi8/pycausalsim) - the Python framework for causal discovery through simulation.

## 🚀 Live Demo

Try it online: [pycausalsim.streamlit.app](https://pycausalsim.streamlit.app) *(after deployment)*

## Features

- **🎯 Causal Simulator** - Discover causal graphs and simulate interventions
- **📊 Marketing Attribution** - Shapley values for true channel attribution
- **🧪 A/B Test Analysis** - Doubly-robust estimation with heterogeneity analysis
- **👥 Uplift Modeling** - Segment users into persuadables, lost causes, etc.

## Deploy to Streamlit Cloud (Free)

### Step 1: Fork/Clone

```bash
git clone https://github.com/Bodhi8/pycausalsim-demo.git
cd pycausalsim-demo
```

Or create a new repo with these files.

### Step 2: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/pycausalsim-demo.git
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file: `app.py`
6. Click "Deploy"

Your app will be live at: `https://your-app-name.streamlit.app`

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Embed on Your Website

Add this iframe to your website:

```html
<iframe
  src="https://pycausalsim.streamlit.app/?embedded=true"
  width="100%"
  height="800"
  frameborder="0"
></iframe>
```

Or link directly:

```html
<a href="https://pycausalsim.streamlit.app" target="_blank">
  Try PyCausalSim Demo →
</a>
```

## Project Structure

```
pycausalsim-demo/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .streamlit/
    └── config.toml    # Streamlit theme config
```

## Customization

Edit `.streamlit/config.toml` to customize the theme:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## License

MIT License - see [PyCausalSim](https://github.com/Bodhi8/pycausalsim) for details.

## Author

Built by [Brian Curry](https://vector1.ai) | [GitHub](https://github.com/Bodhi8) | [Medium](https://medium.com/@briancurry)
