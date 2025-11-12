# Multi-stage build for AI-Ops Incident Responder
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt


# Final stage - minimal runtime image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application files
COPY agent_handoff.py .
COPY runbooks.json .
COPY mcp_servers_config.json .
COPY .env .

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
