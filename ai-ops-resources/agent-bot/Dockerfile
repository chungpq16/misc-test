# AI-Ops Incident Responder Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY agent_handoff.py .
COPY runbooks.json .
COPY mcp_servers_config.json .

# Create non-root user for security
RUN useradd -m -u 1000 aiops && \
    chown -R aiops:aiops /app

# Switch to non-root user
USER aiops

# Set Python to run in unbuffered mode for real-time logs
ENV PYTHONUNBUFFERED=1

# Health check - verify process is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD pgrep -f agent_handoff.py || exit 1

# Run the application
CMD ["python", "-u", "agent_handoff.py"]
