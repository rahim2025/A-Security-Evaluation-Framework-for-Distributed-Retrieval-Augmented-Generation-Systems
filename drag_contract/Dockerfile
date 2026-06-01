FROM node:20-slim

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python3 -> python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory to /app (project root)
WORKDIR /app

# Copy drag_contract package files first for better layer caching
COPY drag_contract/package.json drag_contract/package-lock.json ./drag_contract/

# Install Node.js dependencies
WORKDIR /app/drag_contract
RUN npm install

# Copy rest of drag_contract directory
WORKDIR /app
COPY drag_contract/ ./drag_contract/

# Copy Python client requirements and install dependencies
COPY drag_python_client/requirements.txt ./drag_python_client/
RUN pip3 install --no-cache-dir --break-system-packages -r drag_python_client/requirements.txt

# Copy drag_python_client directory
COPY drag_python_client/ ./drag_python_client/

# Copy drag_data_source/configs directory (needed for test_default_sources)
COPY drag_data_source/configs/ ./drag_data_source/configs/

# Create entrypoint script
COPY drag_contract/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create healthcheck script
COPY drag_contract/healthcheck.sh /healthcheck.sh
RUN chmod +x /healthcheck.sh

# Expose Hardhat node port
EXPOSE 8545

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
