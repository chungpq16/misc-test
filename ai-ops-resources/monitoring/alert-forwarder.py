from fastapi import FastAPI, Request, HTTPException
import pika
import os
import json
import logging
from datetime import datetime
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/alert-forwarder.log')
    ]
)
logger = logging.getLogger(__name__)

# RabbitMQ configuration
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE_NAME = os.getenv("QUEUE_NAME", "test-queue")

class RabbitMQConnection:
    def __init__(self):
        self.connection = None
        self.channel = None
        
    def connect(self):
        """Establish connection to RabbitMQ."""
        try:
            logger.info(f"Attempting to connect to RabbitMQ: {RABBITMQ_URL}")
            params = pika.URLParameters(RABBITMQ_URL)
            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()
            
            # Declare queue (idempotent: safe if it already exists)
            self.channel.queue_declare(queue=QUEUE_NAME, durable=True)
            logger.info(f"Successfully connected to RabbitMQ. Queue: {QUEUE_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            logger.debug(f"RabbitMQ URL: {RABBITMQ_URL}, Queue: {QUEUE_NAME}")
            raise
    
    def publish_message(self, message):
        """Publish a message to RabbitMQ."""
        try:
            logger.debug(f"Attempting to publish message: {json.dumps(message, indent=2)}")
            
            if not self.connection or self.connection.is_closed:
                logger.warning("RabbitMQ connection is closed, reconnecting...")
                self.connect()
                
            message_body = json.dumps(message).encode("utf-8")
            logger.debug(f"Message body size: {len(message_body)} bytes")
            
            self.channel.basic_publish(
                exchange="",
                routing_key=QUEUE_NAME,
                body=message_body,
                properties=pika.BasicProperties(delivery_mode=2)  # Make message persistent
            )
            logger.info(f"Successfully published message to queue '{QUEUE_NAME}' - Alert: {message.get('alertname', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            logger.debug(f"Failed message content: {json.dumps(message, indent=2)}")
            raise
    
    def close(self):
        """Close RabbitMQ connection."""
        try:
            if self.connection and not self.connection.is_closed:
                logger.info("Closing RabbitMQ connection...")
                self.connection.close()
                logger.info("RabbitMQ connection closed successfully")
            else:
                logger.debug("RabbitMQ connection already closed or not established")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection: {e}")

# Global RabbitMQ connection
rabbitmq_conn = RabbitMQConnection()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Alert Forwarder application...")
    logger.info(f"Configuration - RabbitMQ URL: {RABBITMQ_URL}, Queue: {QUEUE_NAME}")
    try:
        rabbitmq_conn.connect()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Alert Forwarder application...")
    try:
        rabbitmq_conn.close()
        logger.info("Application shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

app = FastAPI(lifespan=lifespan)

@app.post("/alerts")
async def receive_alert(request: Request):
    """Receive alerts and forward them to RabbitMQ using AMQP."""
    request_id = id(request)  # Simple request ID for tracking
    logger.info(f"[REQ-{request_id}] Received alert webhook request")
    
    try:
        # Get client info for logging
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        content_type = request.headers.get("content-type", "unknown")
        
        logger.info(f"[REQ-{request_id}] Client: {client_host}, User-Agent: {user_agent}, Content-Type: {content_type}")
        
        alert_body = await request.json()
        logger.debug(f"[REQ-{request_id}] Raw alert body: {json.dumps(alert_body, indent=2)}")
        
        alerts_count = len(alert_body.get("alerts", []))
        logger.info(f"[REQ-{request_id}] Processing {alerts_count} alerts")
        
        if alerts_count == 0:
            logger.warning(f"[REQ-{request_id}] No alerts found in request body")
            return {"status": "ok", "forwarded": 0, "message": "No alerts to process"}
        
        # Optional: reduce / normalize payload before pushing to queue
        messages = []
        for i, alert in enumerate(alert_body.get("alerts", [])):
            logger.debug(f"[REQ-{request_id}] Processing alert {i+1}/{alerts_count}: {alert.get('labels', {}).get('alertname', 'unknown')}")
            
            msg = {
                "alertname": alert["labels"].get("alertname"),
                "severity": alert["labels"].get("severity", "unknown"),
                "status": alert.get("status"),
                "summary": alert.get("annotations", {}).get("summary"),
                "description": alert.get("annotations", {}).get("description"),
                "startsAt": alert.get("startsAt"),
                "endsAt": alert.get("endsAt"),
                "labels": alert.get("labels", {}),
                "annotations": alert.get("annotations", {}),
                "generatorURL": alert.get("generatorURL"),
                "fingerprint": alert.get("fingerprint"),
                # Keep original full payload for GenAI context if needed
                "raw": alert_body,
                # Add tracking metadata
                "request_id": request_id,
                "processed_at": datetime.now().isoformat(),
                "client_host": client_host
            }
            messages.append(msg)
            logger.debug(f"[REQ-{request_id}] Normalized alert {i+1}: {msg.get('alertname')} (severity: {msg.get('severity')})")

        # Publish each message to RabbitMQ
        successful_publishes = 0
        for i, msg in enumerate(messages):
            try:
                rabbitmq_conn.publish_message(msg)
                successful_publishes += 1
                logger.info(f"[REQ-{request_id}] Published alert {i+1}/{len(messages)}: {msg.get('alertname', 'unknown')}")
            except Exception as publish_error:
                logger.error(f"[REQ-{request_id}] Failed to publish alert {i+1}/{len(messages)}: {publish_error}")
                # Continue with other messages instead of failing completely

        logger.info(f"[REQ-{request_id}] Completed processing: {successful_publishes}/{len(messages)} alerts published successfully")
        
        return {
            "status": "ok", 
            "forwarded": successful_publishes,
            "total_alerts": len(messages),
            "request_id": request_id
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"[REQ-{request_id}] Invalid JSON in request body: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"[REQ-{request_id}] Error processing alerts: {e}")
        logger.exception(f"[REQ-{request_id}] Full traceback:")
        raise HTTPException(status_code=500, detail=f"Error processing alerts: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
