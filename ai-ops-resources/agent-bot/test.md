alert_message = "HIGH: Nginx deployment in default namespace experiencing high latency (500ms avg response time). CPU usage at 85% and increasing. Current replica count appears insufficient for traffic load. Scale up required to handle overload."


alert_message = "CRITICAL: Intermittent application timeouts affecting 25% of user requests across multiple microservices. Database connection pool exhaustion detected with unstable connections. API gateway reporting degraded performance. Network latency spikes observed between services. This is a complex multi-component issue requiring expert root cause analysis and efficiency planning. Do NOT attempt automated remediation."


import pika
import json

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='alert-queue', durable=True)

# Scenario 1: Performance Issues - Safe for Automation
alert1 = {
  "severity": "HIGH",
  "source": "prometheus-monitoring",
  "timestamp": "2025-11-12T10:30:00Z",
  "message": "HIGH: Nginx deployment in default namespace experiencing high latency (500ms avg response time) and slow response time. Application shows performance degradation with CPU usage at 85% and increasing. Memory utilization trending upward."
}

# Scenario 2: Network Issues - Requires Investigation
alert2 = {
  "severity": "CRITICAL",
  "source": "application-monitoring",
  "timestamp": "2025-11-12T10:35:00Z",
  "message": "CRITICAL: Network connectivity issues affecting multiple microservices. Intermittent connection timeouts and service unavailable errors observed. Database connection failures with 'connection refused' errors occurring. API gateway experiencing network latency to unreachable backend services. DNS resolution failures detected across service mesh."
}

# Scenario 3: Application Stuck - Safe for Automation (Restart)
alert3 = {
  "severity": "HIGH",
  "source": "application-monitoring",
  "timestamp": "2025-11-12T10:40:00Z",
  "message": "HIGH: Nginx deployment in default namespace appears unresponsive and stuck. Application not responding to health checks. Pods showing hanging behavior with no request processing. Service appears frozen despite being in Running state. Immediate restart required to recover service."
}