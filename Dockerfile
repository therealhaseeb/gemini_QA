FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies and clean up
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment and install Python dependencies
COPY requirements.txt /app/
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Runtime image
FROM python:3.11-slim AS runtime

# Create a non-root user and group
RUN groupadd --system appgroup \
    && useradd --system --create-home --gid appgroup --shell /bin/bash appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only application code
COPY app.py /app/
COPY requirements.txt /app/

# Set permissions and switch to non-root user
RUN chown -R appuser:appgroup /app /opt/venv
USER appuser

EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
