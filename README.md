# Comment Toxicity Detection Web App

This project provides a complete pipeline for comment toxicity classification, including scripts to fine-tune a Hugging Face `transformers` model (specifically DeBERTa) and a Flask-based web application to serve the trained model. Users can check comment toxicity via a web interface, and the application is instrumented with Prometheus for monitoring API performance.

## Features

- Web interface for real-time toxicity checking.
- Flask backend serving a PyTorch-based Hugging Face model.
- Automatic CPU/GPU detection (defaults to CPU if no GPU is available).
- Prometheus metrics endpoint (`/metrics`) for API request counts and latency.
- Docker Compose setup for running the API along with Prometheus and Grafana for monitoring.
- Script to fine-tune a DeBERTa model on the Jigsaw toxicity dataset.

## Prerequisites

- Python 3.8+
- `pip` (Python package installer)
- Git (for cloning the repository)

### Docker and Docker Compose (for Production/Full Stack)

To run the full monitoring stack, Docker is required. Here are the installation instructions for a typical Debian-based Linux server (e.g., Ubuntu).

1.  **Install Docker Engine:**
    ```bash
    # Update package index and install prerequisites
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg

    # Add Docker's official GPG key
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    # Set up the repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ```

2.  **Manage Docker as a non-root user (Optional but recommended):**
    ```bash
    sudo groupadd docker
    sudo usermod -aG docker $USER
    # Log out and log back in for this to take effect.
    ```

3.  **Verify Docker Installation:**
    ```bash
    docker --version
    docker compose version
    ```

## Installation and Setup (CPU-Only Server)

These instructions guide you through setting up and running the Flask application directly on a server, utilizing its CPU for model inference.

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and Activate a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    *(On Windows, use `venv\Scripts\activate`)*

3.  **Install Dependencies:**
    Navigate to the project root directory (where `src/requirements.txt` is located) and install the required Python packages.
    ```bash
    pip install -r src/requirements.txt
    ```

4.  **Place Model Files:**
    The application expects the trained Hugging Face `transformers` model files (e.g., `pytorch_model.bin`, `config.json`, `tokenizer_config.json`, etc.) to be located in a directory named `model` in the project root.
    ```
    <project-root>/
    ├── model/                 # Your model files go here
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   └── ... (other tokenizer/model files)
    ├── src/
    └── ...
    ```
    Alternatively, you can specify a custom model directory path by setting the `MODEL_DIR` environment variable before running the application:
    ```bash
    export MODEL_DIR=/path/to/your/model_directory
    ```

## Running the Application (CPU-Only Server)

1.  **Start the Flask Application:**
    Ensure your virtual environment is activated and you are in the project root directory.
    ```bash
    python src/app.py
    ```
    The application will start, and since it's a CPU-only server (or no CUDA-enabled GPU is detected), it will automatically use the CPU for model inference.

2.  **Access the Web UI:**
    Open your web browser and navigate to `http://localhost:5000` (or `http://<your-server-ip>:5000` if running on a remote server and the port is open).

3.  **Access Metrics Endpoint:**
    To see the Prometheus metrics, navigate to `http://localhost:5000/metrics`.

## Running with Full Monitoring Stack (Docker Compose)

This method uses Docker and Docker Compose to run the Flask API, Prometheus, and Grafana together in containers. This is suitable for a more robust deployment and for utilizing the monitoring capabilities.

1.  **Ensure Docker is Installed and Running.**

2.  **Place Model Files:**
    Make sure your model files are in the `./model/` directory at the project root as described in step 4 of the CPU-only installation. The `docker-compose.yml` file is configured to mount this directory into the API container.

3.  **Build and Run with Docker Compose:**
    From the project root directory (where `docker-compose.yml` is located):
    ```bash
    docker compose up --build
    ```
    This command will build the API image and start all services (`api`, `prometheus`, `grafana`).

4.  **Access Services:**
    -   **Web App UI:** `http://localhost:5000`
    -   **API Metrics:** `http://localhost:5000/metrics`
    -   **Prometheus:** `http://localhost:9090`
    -   **Grafana:** `http://localhost:3000` (default login: `admin` / `admin`)

5.  **Configure Grafana Data Source:**
    -   Log in to Grafana.
    -   Go to "Connections" (or the gear icon -> "Data Sources").
    -   Click "Add data source".
    -   Select "Prometheus".
    -   Set the "Prometheus server URL" to `http://prometheus:9090`.
    -   Click "Save & test". You should see a success message.
    -   You can now create dashboards in Grafana using the `tox_api_requests_total` and `tox_api_latency_seconds` metrics (and others).

## Training the Model

The model served by this application is a fine-tuned **DeBERTa (`microsoft/deberta-v3-base`)** model. It was trained for multi-label text classification on the [Jigsaw Toxicity Prediction](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset.

To run the training process yourself:

1.  **Ensure all dependencies are installed:**
    Make sure you have installed the packages from `src/requirements.txt` into your virtual environment.

2.  **Download the dataset:**
    The training script expects the Jigsaw dataset files to be in the `data/jigsaw/` directory.

3.  **Run the training script:**
    From the project root directory, run the following command:
    ```bash
    python -m src.train
    ```
    This will start the fine-tuning process. Once complete, the trained model and tokenizer files will be saved to the `model/` directory, ready to be served by the Flask application.

## Production Deployment with Nginx Reverse Proxy

When running this stack on a production server, it's best practice to use a reverse proxy like Nginx to manage incoming traffic. This allows you to run multiple web services on a single server, handle SSL termination (HTTPS), and serve everything from the standard web ports (80/443).

The following configuration will set up Nginx to:
- Serve the **Toxicity App** on the main domain (`/`).
- Serve **Prometheus** on the `/prometheus/` subpath.
- Serve **Grafana** on the `/grafana/` subpath.

### 1. Update Docker Compose for Grafana

For Grafana to work correctly when served from a subpath (e.g., `/grafana/`), you must update its configuration in the `docker-compose.yml` file. You need to set environment variables to inform Grafana of its public URL.

Also, the default port inside the Grafana container is `3000`. Your `docker-compose.yml` should map your desired host port (e.g., `3100`) to the container's port `3000`.

Make sure the `grafana` service in your `docker-compose.yml` looks like this (remember to replace `your_domain.com`):

```yaml
  grafana:
    image: grafana/grafana:9.5.3
    ports:
      - "3100:3000" # Expose on host port 3100, maps to container port 3000
    volumes:
      - ./monitoring/grafana:/var/lib/grafana
    environment:
      - GF_SERVER_ROOT_URL=http://your_domain.com/grafana
      - GF_SERVER_SERVE_FROM_SUB_PATH=true
```

### 2. Create the Nginx Configuration File

Create a new file in `/etc/nginx/sites-available/`, for example, `toxicity-app.conf`. **Remember to replace `your_domain.com` with your actual domain name.**

```bash
sudo nano /etc/nginx/sites-available/toxicity-app.conf
```

Paste the following configuration into the file:

```nginx
server {
    listen 80;
    server_name your_domain.com;

    # Proxy for the main Flask App
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Proxy for Prometheus
    # Note the trailing slash on both location and proxy_pass
    location /prometheus/ {
        proxy_pass http://localhost:9090/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Proxy for Grafana
    # Note the trailing slash on both location and proxy_pass
    location /grafana/ {
        # Proxy to the HOST port you exposed in docker-compose.yml
        proxy_pass http://localhost:3100/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. Enable the Site and Test Configuration

Create a symbolic link from `sites-available` to `sites-enabled` to activate the configuration:

```bash
sudo ln -s /etc/nginx/sites-available/toxicity-app.conf /etc/nginx/sites-enabled/
```

Test your Nginx configuration for syntax errors:

```bash
sudo nginx -t
```

If the test is successful, reload Nginx to apply the changes:

```bash
sudo systemctl reload nginx
```

You should now be able to access your services at `http://your_domain.com`, `http://your_domain.com/prometheus/`, and `http://your_domain.com/grafana/`.

## Model Information

The application is designed to load a pre-trained sequence classification model compatible with Hugging Face `transformers` (e.g., a fine-tuned BERT, RoBERTa, DistilBERT, etc.). The `app.py` script uses `AutoTokenizer.from_pretrained()` and `AutoModelForSequenceClassification.from_pretrained()`.

The script automatically handles device placement (CPU/GPU). On a CPU-only server, it will default to using the CPU for inference.
